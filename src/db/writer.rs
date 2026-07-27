// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! The single, serialized writer: `put` (upsert), `delete`, and `commit`.
//!
//! Single-writer is enforced at compile time — [`crate::db::Store::writer`]
//! hands out a `Writer` holding the store's writer mutex guard, so only one can
//! exist at a time. This sidesteps the engine's shared-mutation-buffer hazard
//! (DB-layer plan §8): there is exactly one logical mutation path.
//!
//! Lowering (plan §7): a `put` is `delete_frame`/`delete_vector` of the prior
//! version (when the key exists) → `put_frame` (key→uri) → `index_frame_text`
//! → `put_vector` per field. `commit` first drives **deferred calibration**
//! (calibrate a vector space's codec from the first N staged vectors, register
//! it, flush the staged vectors) and then runs the engine's avalanche commit
//! through a `WriteConnection` so the published snapshot advances.
//!
//! ## Lock discipline
//! Three locks are involved: `schema`, the engine's `inner` `RwLock`, and
//! `identity`. They are always taken one at a time in the order
//! `schema → inner → identity`; no two are held simultaneously, so there is no
//! deadlock surface.

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::{ArcMutexGuard, Mutex, MutexGuard, RawMutex};

use crate::concurrency::database::Database;
use crate::db::collection::Field;
use crate::db::identity::{Entry, IdentityIndex};
use crate::db::loader::Bulk;
use crate::db::partition::Partitioned;
use crate::db::record::{Key, Record};
use crate::db::schema::CodecSpec;
use crate::db::schema_doc;
use crate::db::space::{
    CollectionRecord, Metric, SchemaRegistry, VecState, register_codec_for,
    register_embedding_space,
};
use crate::error::{Error, Result};
use crate::file::{CommitOutcome, PutFrame, PutVector};
use crate::format::catalog::FrameRole;
use crate::format::{CollectionId, EmbeddingSpaceId, FrameId, TextSpaceId, VectorId};

/// A vector destined for a deferred-calibration space, held until the codec is
/// calibrated at `commit`.
struct StagedVector {
    space_name: String,
    collection: CollectionId,
    key_bytes: Vec<u8>,
    owner_frame: FrameId,
    values: Vec<f32>,
}

/// The state and all the real logic of the serialized writer, independent of
/// how the single-writer lock guard is held. Both [`Writer`] (borrowed guard)
/// and [`OwnedWriter`] (owned `Arc` guard) hold one of these and delegate to it;
/// the guard lives on the wrapper, not here.
pub(crate) struct WriterCore {
    db: Arc<Database>,
    schema: Arc<Mutex<SchemaRegistry>>,
    identity: Arc<Mutex<IdentityIndex>>,
    staged: Vec<StagedVector>,
}

impl WriterCore {
    fn new(
        db: Arc<Database>,
        schema: Arc<Mutex<SchemaRegistry>>,
        identity: Arc<Mutex<IdentityIndex>>,
    ) -> Self {
        Self {
            db,
            schema,
            identity,
            staged: Vec::new(),
        }
    }

    /// Create-or-update a record under `key` in `coll`. If the key already
    /// exists (and is committed), its prior frame and vectors are tombstoned
    /// and the key is remapped to the new frame — synthesizing `update` over an
    /// append-only engine. Pending writes land at the next [`Writer::commit`].
    fn put(&mut self, coll: &str, key: Key, rec: Record<'_>) -> Result<()> {
        let key_bytes = key.encode();

        // ---- Phase 1: resolve the put plan from the schema (lock, then drop).
        let plan = {
            let schema = self.schema.lock();
            self.resolve_plan(&schema, coll, &rec)?
        };

        // ---- Phase 2: inspect identity for the existing version + parent.
        let (existing, parent_frame) = {
            let identity = self.identity.lock();
            let existing = identity.get(plan.collection_id, &key_bytes).cloned();
            let parent_frame = match &rec.parent {
                Some(pk) => {
                    let pk_bytes = pk.encode();
                    Some(
                        identity
                            .get(plan.collection_id, &pk_bytes)
                            .map(|e| e.frame_id)
                            .ok_or_else(|| {
                                Error::Format(format!(
                                    "put: parent key not found in collection '{coll}'"
                                ))
                            })?,
                    )
                }
                None => None,
            };
            (existing, parent_frame)
        };

        // An update whose prior version still has uncommitted vectors cannot be
        // expressed (the engine refuses to tombstone an uncommitted vector).
        if let Some(prev) = &existing
            && !prev.committed
            && !prev.vectors.is_empty()
        {
            return Err(Error::Unsupported(
                "put: cannot update a key whose vectors are not yet committed; commit() first"
                    .into(),
            ));
        }

        let text = rec.text_value();
        let payload: &[u8] = text.map(str::as_bytes).unwrap_or(&[]);

        // ---- Phase 3: engine mutations under the inner write guard.
        //
        // The new version is appended *before* the prior version is tombstoned,
        // so a mid-way error leaves the prior version intact and committable
        // (the orphaned partial new frame is never referenced by identity and
        // is reclaimed at compaction). On any error here the caller must
        // discard the batch (drop without `commit`).
        let mut ready_vectors: Vec<(EmbeddingSpaceId, VectorId)> =
            Vec::with_capacity(plan.ready.len());
        let frame_id = {
            let mut valise = self.db.inner.write();

            let frame_id = valise.put_frame(PutFrame {
                collection_id: plan.collection_id,
                role: if parent_frame.is_some() {
                    FrameRole::Chunk
                } else {
                    FrameRole::Document
                },
                payload,
                created_at: rec.created_at,
                updated_at: None,
                parent_frame_id: parent_frame,
                chunk_index: None,
                chunk_count: None,
                uri: Some(&key_bytes),
            })?;

            if let (Some(ts_id), Some(_)) = (plan.text_space_id, text) {
                valise.index_frame_text(frame_id, ts_id)?;
            }

            for (space_id, values) in &plan.ready {
                let vid = valise.put_vector(PutVector {
                    owner_frame_id: frame_id,
                    embedding_space_id: *space_id,
                    values,
                })?;
                ready_vectors.push((*space_id, vid));
            }

            // Tombstone the prior version last.
            if let Some(prev) = &existing {
                if prev.committed {
                    for (_, vid) in &prev.vectors {
                        valise.delete_vector(*vid)?;
                    }
                }
                valise.delete_frame(prev.frame_id)?;
            }

            frame_id
        };

        // Drop any pre-calibration staged vectors that pointed at the prior
        // frame we just tombstoned, so `flush_deferred` never re-puts onto a
        // tombstoned frame (the engine rejects that, failing the whole commit).
        // When `prev` was committed it owns no staged vectors, so this is a
        // no-op there.
        if let Some(prev) = &existing {
            self.staged.retain(|s| s.owner_frame != prev.frame_id);
        }

        // Stage deferred-space vectors for calibration at commit.
        for (space_name, values) in &plan.deferred {
            self.staged.push(StagedVector {
                space_name: space_name.clone(),
                collection: plan.collection_id,
                key_bytes: key_bytes.clone(),
                owner_frame: frame_id,
                values: values.to_vec(),
            });
        }

        // ---- Phase 4: remap identity. `insert` overwrites any prior entry for
        // this key, so no separate `remove` is needed.
        {
            let mut identity = self.identity.lock();
            identity.insert(
                plan.collection_id,
                key_bytes,
                Entry {
                    frame_id,
                    vectors: ready_vectors,
                    committed: false,
                },
            );
        }

        Ok(())
    }

    /// Create a record under a layer-generated key, returning it.
    fn put_auto(&mut self, coll: &str, rec: Record<'_>) -> Result<Key> {
        let key = Key::Str(uuid::Uuid::new_v4().to_string());
        self.put(coll, key.clone(), rec)?;
        Ok(key)
    }

    /// Route a record into its partition (lazily creating the physical
    /// collection, bound to the partitioned shape) and `put` it there.
    ///
    /// Ingest order: the engine time index requires `created_at` to be
    /// non-decreasing in commit order, so ingest oldest-first within a commit
    /// (natural for append-as-time-flows memory workloads). Backfilling
    /// out-of-order timestamps within one commit errors at `commit`.
    fn put_into(&mut self, p: &Partitioned, key: Key, rec: Record<'_>) -> Result<()> {
        let name = p.route(&rec, now_secs());
        self.ensure_collection_with_shape(&name, p)?;
        self.put(&name, key, rec)
    }

    /// Create-or-open a partition's physical collection, bind the shape for
    /// this process, and persist the partition's schema doc. Goes through
    /// the shared schema-guarded create path; the doc binds the *base*'s
    /// `~auto/{base}/{field}` spaces (partitions share them by design), so
    /// reopen restores the partition with no re-declaration.
    fn ensure_collection_with_shape(&self, name: &str, p: &Partitioned) -> Result<()> {
        let mut schema = self.schema.lock();
        if schema
            .collections
            .get(name)
            .is_some_and(|rec| rec.shape.is_some())
        {
            return Ok(());
        }
        let doc = schema_doc::SchemaDoc::from_schema(p.base(), p.declared_schema())?;
        let plan = {
            let valise = self.db.inner.read();
            schema_doc::plan_sync(&valise, name, &doc)?
        };
        let collection_id = crate::db::ensure_collection(&self.db, &mut schema, name)?;
        {
            let mut valise = self.db.inner.write();
            schema_doc::apply_sync(&mut valise, name, &doc, &plan)?;
        }
        schema.collections.insert(
            name.to_owned(),
            CollectionRecord {
                collection_id,
                shape: Some(p.shape().clone()),
                doc: Some(doc),
            },
        );
        Ok(())
    }

    /// Tombstone the record under `key` and unmap it. Returns whether a record
    /// existed. Like `put`, an uncommitted-with-vectors record must be
    /// committed before it can be removed.
    fn delete(&mut self, coll: &str, key: &Key) -> Result<bool> {
        let key_bytes = key.encode();
        let collection_id = {
            let schema = self.schema.lock();
            schema
                .collections
                .get(coll)
                .map(|c| c.collection_id)
                .ok_or_else(|| Error::Format(format!("delete: unknown collection '{coll}'")))?
        };

        let entry = {
            let identity = self.identity.lock();
            identity.get(collection_id, &key_bytes).cloned()
        };
        let Some(entry) = entry else {
            return Ok(false);
        };
        if !entry.committed && !entry.vectors.is_empty() {
            return Err(Error::Unsupported(
                "delete: cannot remove a key whose vectors are not yet committed; commit() first"
                    .into(),
            ));
        }

        {
            let mut valise = self.db.inner.write();
            if entry.committed {
                for (_, vid) in &entry.vectors {
                    valise.delete_vector(*vid)?;
                }
            }
            valise.delete_frame(entry.frame_id)?;
        }

        // A pure-deferred record's vectors live in `self.staged` (not
        // `entry.vectors`), so the frame we just tombstoned may still have
        // staged vectors pinned to it — drop them so the next `flush_deferred`
        // doesn't re-put onto a tombstoned frame.
        self.staged.retain(|s| s.owner_frame != entry.frame_id);

        self.identity.lock().remove(collection_id, &key_bytes);
        Ok(true)
    }

    /// Commit all pending mutations. Drives deferred calibration first, then
    /// runs the engine's avalanche commit (which republishes the snapshot).
    fn commit(&mut self) -> Result<CommitOutcome> {
        if !self.staged.is_empty() {
            self.flush_deferred()?;
        }
        let outcome = {
            let mut wc = self.db.writer();
            wc.commit()?
        };
        self.identity.lock().mark_all_committed();
        Ok(outcome)
    }

    /// Calibrate every deferred vector space that has staged vectors, register
    /// its codec + embedding space, then flush all staged vectors.
    fn flush_deferred(&mut self) -> Result<()> {
        // Distinct space names, first-seen order.
        let mut names: Vec<String> = Vec::new();
        for s in &self.staged {
            if !names.iter().any(|n| n == &s.space_name) {
                names.push(s.space_name.clone());
            }
        }

        struct Calib {
            dim: u32,
            metric: Metric,
            sample_target: usize,
            codec: CodecSpec,
        }
        enum Plan {
            Ready(EmbeddingSpaceId),
            Calib(Calib),
        }

        // Resolve each space's current state from the schema.
        let mut plans: HashMap<String, Plan> = HashMap::with_capacity(names.len());
        {
            let schema = self.schema.lock();
            for name in &names {
                let entry = schema.vector_spaces.get(name).ok_or_else(|| {
                    Error::Format(format!("commit: staged vectors for unknown space '{name}'"))
                })?;
                let plan = match &entry.state {
                    VecState::Ready { space_id } => Plan::Ready(*space_id),
                    VecState::Deferred {
                        sample_target,
                        codec,
                    } => Plan::Calib(Calib {
                        dim: entry.dim,
                        metric: entry.metric,
                        sample_target: *sample_target,
                        codec: codec.clone(),
                    }),
                };
                plans.insert(name.clone(), plan);
            }
        }

        // Calibrate + register + put every staged vector.
        let mut resolved: HashMap<String, EmbeddingSpaceId> = HashMap::with_capacity(names.len());
        let mut newly_ready: Vec<(String, EmbeddingSpaceId)> = Vec::new();
        let mut pushes: Vec<(CollectionId, Vec<u8>, EmbeddingSpaceId, VectorId)> =
            Vec::with_capacity(self.staged.len());
        {
            let mut valise = self.db.inner.write();
            for name in &names {
                match plans.get(name).expect("BUG: plan present for staged space") {
                    Plan::Ready(space_id) => {
                        resolved.insert(name.clone(), *space_id);
                    }
                    Plan::Calib(c) => {
                        let sample: Vec<Vec<f32>> = self
                            .staged
                            .iter()
                            .filter(|s| &s.space_name == name)
                            .take(c.sample_target)
                            .map(|s| s.values.clone())
                            .collect();
                        if sample.is_empty() {
                            continue;
                        }
                        let codec = register_codec_for(&mut valise, c.dim, &c.codec, &sample)?;
                        let space_id =
                            register_embedding_space(&mut valise, name, c.dim, c.metric, codec)?;
                        resolved.insert(name.clone(), space_id);
                        newly_ready.push((name.clone(), space_id));
                    }
                }
            }

            for s in &self.staged {
                let Some(&space_id) = resolved.get(&s.space_name) else {
                    continue;
                };
                let vid = valise.put_vector(PutVector {
                    owner_frame_id: s.owner_frame,
                    embedding_space_id: space_id,
                    values: &s.values,
                })?;
                pushes.push((s.collection, s.key_bytes.clone(), space_id, vid));
            }
        }

        // Promote newly calibrated spaces to Ready.
        if !newly_ready.is_empty() {
            let mut schema = self.schema.lock();
            for (name, space_id) in newly_ready {
                if let Some(e) = schema.vector_spaces.get_mut(&name) {
                    e.state = VecState::Ready { space_id };
                }
            }
        }

        // Attach the flushed vector ids to their identity entries.
        {
            let mut identity = self.identity.lock();
            for (coll, key_bytes, space_id, vid) in pushes {
                identity.push_vector(coll, &key_bytes, space_id, vid);
            }
        }

        self.staged.clear();
        Ok(())
    }

    /// Resolve a `put` against the shape bound to `coll`: validate fields,
    /// split vector fields into ready (already-calibrated) and deferred, and
    /// find the text space id.
    fn resolve_plan<'a>(
        &self,
        schema: &SchemaRegistry,
        coll: &str,
        rec: &Record<'a>,
    ) -> Result<PutPlan<'a>> {
        let record = schema
            .collections
            .get(coll)
            .ok_or_else(|| Error::Format(format!("put: unknown collection '{coll}'")))?;
        let shape = record.shape.as_ref().ok_or_else(|| {
            Error::Format(format!(
                "put: collection '{coll}' has no shape in this process; call store.collection(name, shape) first"
            ))
        })?;
        let collection_id = record.collection_id;

        // Validate every supplied field against the shape.
        for (fname, value) in &rec.fields {
            let field = shape.field(fname).ok_or_else(|| {
                Error::Format(format!(
                    "put: field '{fname}' is not in collection '{coll}'"
                ))
            })?;
            match (field, value) {
                (Field::Text(_), crate::db::record::Value::Text(_)) => {}
                (Field::Vector(space), crate::db::record::Value::Vector(v)) => {
                    if let Some(dim) = space.dim()
                        && dim as usize != v.len()
                    {
                        return Err(Error::Format(format!(
                            "put: field '{fname}' expects dim {dim}, got {}",
                            v.len()
                        )));
                    }
                    // Reject NaN/±Inf at the offending put, so a bad vector
                    // can't poison the deferred commit-time encode (which would
                    // fail the whole batch with no record context).
                    if !all_finite(v) {
                        return Err(Error::Format(format!(
                            "put: field '{fname}' in collection '{coll}' has non-finite values (NaN/Inf)"
                        )));
                    }
                }
                _ => {
                    return Err(Error::Format(format!(
                        "put: field '{fname}' kind does not match the shape"
                    )));
                }
            }
        }

        // Text space (if the shape has a text field and the record set it).
        let text_space_id: Option<TextSpaceId> = match (shape.text_field(), rec.text_value()) {
            (Some((_, space)), Some(_)) => {
                Some(schema.text_space_id(space.name()).ok_or_else(|| {
                    Error::Format(format!("put: text space '{}' not registered", space.name()))
                })?)
            }
            _ => None,
        };

        // Split vector fields by calibration readiness.
        let mut ready: Vec<(EmbeddingSpaceId, &'a [f32])> = Vec::new();
        let mut deferred: Vec<(String, &'a [f32])> = Vec::new();
        for (fname, values) in rec.vector_values() {
            let Some(Field::Vector(space)) = shape.field(fname) else {
                // Already validated above; defensive.
                return Err(Error::Format(format!(
                    "put: field '{fname}' is not a vector field"
                )));
            };
            let name = space.name();
            let entry = schema.vector_spaces.get(name).ok_or_else(|| {
                Error::Format(format!("put: vector space '{name}' not registered"))
            })?;
            match entry.state {
                VecState::Ready { space_id } => ready.push((space_id, values)),
                VecState::Deferred { .. } => deferred.push((name.to_owned(), values)),
            }
        }

        Ok(PutPlan {
            collection_id,
            text_space_id,
            ready,
            deferred,
        })
    }
}

/// The serialized writer borrowing the store's writer lock. See module docs.
///
/// A thin wrapper over `WriterCore`: it holds the borrowed lock guard and
/// delegates every operation to the core. The lifetime ties the writer to the
/// `Store` it came from. For a lock guard with no borrow, see [`OwnedWriter`].
pub struct Writer<'s> {
    _guard: MutexGuard<'s, ()>,
    core: WriterCore,
}

impl<'s> Writer<'s> {
    pub(crate) fn new(
        guard: MutexGuard<'s, ()>,
        db: Arc<Database>,
        schema: Arc<Mutex<SchemaRegistry>>,
        identity: Arc<Mutex<IdentityIndex>>,
    ) -> Self {
        Self {
            _guard: guard,
            core: WriterCore::new(db, schema, identity),
        }
    }

    /// See `WriterCore::put`.
    pub fn put(&mut self, coll: &str, key: impl Into<Key>, rec: Record<'_>) -> Result<()> {
        self.core.put(coll, key.into(), rec)
    }

    /// See `WriterCore::put_auto`.
    pub fn put_auto(&mut self, coll: &str, rec: Record<'_>) -> Result<Key> {
        self.core.put_auto(coll, rec)
    }

    /// See `WriterCore::put_into`.
    pub fn put_into(
        &mut self,
        p: &Partitioned,
        key: impl Into<Key>,
        rec: Record<'_>,
    ) -> Result<()> {
        self.core.put_into(p, key.into(), rec)
    }

    /// See `WriterCore::delete`.
    pub fn delete(&mut self, coll: &str, key: impl Into<Key>) -> Result<bool> {
        self.core.delete(coll, &key.into())
    }

    /// See `WriterCore::commit`.
    pub fn commit(&mut self) -> Result<CommitOutcome> {
        self.core.commit()
    }

    /// A batching loader over this writer (group-commit, default 10k batch).
    ///
    /// Only the borrowed `Writer` exposes `bulk`; the owned variant batches by
    /// looping `put` then a single `commit` (the Python layer's pattern).
    pub fn bulk(&mut self) -> Bulk<'_, 's> {
        Bulk::new(self, 10_000)
    }
}

/// The serialized writer holding the store's writer lock by `Arc` — no borrow
/// of the `Store`, hence no lifetime parameter.
///
/// Mirrors [`Writer`]'s public surface (minus [`Writer::bulk`], which is
/// borrowed-`Writer`-only). Acquire one with [`crate::db::Store::writer_owned`];
/// it blocks until any prior writer drops and releases the single-writer lock
/// when dropped. Useful where a `'static`/owned handle is needed (e.g. an FFI
/// object that cannot carry a Rust lifetime).
pub struct OwnedWriter {
    _guard: ArcMutexGuard<RawMutex, ()>,
    core: WriterCore,
}

impl OwnedWriter {
    pub(crate) fn new(
        guard: ArcMutexGuard<RawMutex, ()>,
        db: Arc<Database>,
        schema: Arc<Mutex<SchemaRegistry>>,
        identity: Arc<Mutex<IdentityIndex>>,
    ) -> Self {
        Self {
            _guard: guard,
            core: WriterCore::new(db, schema, identity),
        }
    }

    /// See `WriterCore::put`.
    pub fn put(&mut self, coll: &str, key: impl Into<Key>, rec: Record<'_>) -> Result<()> {
        self.core.put(coll, key.into(), rec)
    }

    /// See `WriterCore::put_auto`.
    pub fn put_auto(&mut self, coll: &str, rec: Record<'_>) -> Result<Key> {
        self.core.put_auto(coll, rec)
    }

    /// See `WriterCore::put_into`.
    pub fn put_into(
        &mut self,
        p: &Partitioned,
        key: impl Into<Key>,
        rec: Record<'_>,
    ) -> Result<()> {
        self.core.put_into(p, key.into(), rec)
    }

    /// See `WriterCore::delete`.
    pub fn delete(&mut self, coll: &str, key: impl Into<Key>) -> Result<bool> {
        self.core.delete(coll, &key.into())
    }

    /// See `WriterCore::commit`.
    pub fn commit(&mut self) -> Result<CommitOutcome> {
        self.core.commit()
    }
}

/// The resolved lowering plan for one `put`.
struct PutPlan<'a> {
    collection_id: CollectionId,
    text_space_id: Option<TextSpaceId>,
    ready: Vec<(EmbeddingSpaceId, &'a [f32])>,
    deferred: Vec<(String, &'a [f32])>,
}

/// Branchless finiteness scan: rejects any `NaN`/`±Inf`. The non-short-circuit
/// `&` (not `&&`) keeps the loop data-independent so it autovectorizes to a
/// SIMD compare+mask reduction (~37 ns for dim=768 on NEON, ~4× the branchy
/// `all(is_finite)`, which a data-dependent early-exit would prevent the
/// compiler from vectorizing). Does NOT validate dtype/semantics — those are
/// the type system's (`&[f32]`) and the dimension check's job.
#[inline]
fn all_finite(v: &[f32]) -> bool {
    v.iter().fold(true, |acc, x| acc & x.is_finite())
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
