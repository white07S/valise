// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Compaction — the GC primitive that reclaims tombstoned bytes (from
//! `delete` / `forget_before`) by streaming the live set into a fresh file and
//! atomically swapping it in (DB-layer plan §7, §13 P4).
//!
//! Mechanics: under the single-writer lock, build a fresh Valise file at a temp
//! path (re-register all schema verbatim, stream **active** frames in
//! frame-id order — which preserves the source's already-valid
//! `(created_at, frame_id)` time ordering and processes parents before
//! children — re-index text, re-ingest vectors), commit+fsync it, `rename`
//! over the original, fsync the directory, reopen, and publish a fresh
//! `{db, schema, identity}` triple with one atomic store.
//!
//! Caveats:
//! - **Byte-reclaim, not crypto-shred.** Reclaimed bytes are gone from the new
//!   file, but the old file's inode lingers until every prior handle
//!   (`Reader`/`Writer`/`Snapshot`) that pinned it drops. True synchronous
//!   right-to-be-forgotten needs crypto-shred (deferred).
//! - **One QAM round-trip.** Vectors are read back as decoded f32 and
//!   re-encoded; with `recalibrate = false` (default) the *same* codec params
//!   are reused, so the re-encode is near-idempotent. `recalibrate = true`
//!   re-fits the codec from the live (reconstructed) sample.
//! - Held handles created before a compact keep reading the old file
//!   (self-consistent, last-committed-wins); call `store.reader()` again for
//!   the new state. `Store`-level ops (`get`/`search`/`writer`) reload the
//!   published state per call and see the new file immediately.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::codec::CodecParams;
use crate::concurrency::database::Database;
use crate::db::identity::IdentityIndex;
use crate::db::schema_doc;
use crate::db::space::SchemaRegistry;
use crate::db::{Handles, Published};
use crate::error::Result;
use crate::file::{
    CreateOptions, EmbeddingSpaceSpec, PutFrame, PutVector, Reconstruct, ValiseFile,
};
use crate::format::catalog::{FrameRole, TextSpaceDesc, VectorStatus};
use crate::format::{CollectionId, EmbeddingSpaceId, FrameId, TextSpaceId};
use crate::io::Durability;

/// Number of live vectors sampled to re-fit a codec when `recalibrate` is set.
const RECALIBRATE_SAMPLE: usize = 50_000;

/// Options for [`crate::db::Store::compact`].
#[derive(Clone, Debug, Default)]
pub struct CompactOptions {
    /// Re-fit each vector space's QAM codec from the live (reconstructed)
    /// sample. Default `false`: reuse the existing params (near-idempotent
    /// re-encode, no recalibration drift). Set `true` after a large deletion
    /// fraction or a distribution shift.
    pub recalibrate: bool,
}

/// Outcome of a compaction.
#[derive(Clone, Debug)]
pub struct CompactReport {
    /// `false` if compaction was a no-op (nothing tombstoned).
    pub compacted: bool,
    pub frames_before: usize,
    pub frames_after: usize,
    pub vectors_before: usize,
    pub vectors_after: usize,
    pub bytes_before: u64,
    pub bytes_after: u64,
}

pub(crate) fn compact(handles: &Handles, opts: &CompactOptions) -> Result<CompactReport> {
    // Exclusive for the whole op (excludes writers AND concurrent compacts).
    // `try_lock` rather than `lock`: the writer lock is not reentrant, so a
    // blocking acquire would *deadlock* if the caller holds a `Writer` (or
    // another compaction) on this thread. Fail fast with a clear error instead.
    let _wguard = handles.writer_lock.try_lock().ok_or_else(|| {
        crate::error::Error::Busy(
            "compact: a writer or another compaction is active; release it first".into(),
        )
    })?;
    let p = handles.current();

    let (
        src_path,
        raw_frames_before,
        raw_frames_after,
        frames_before,
        frames_after,
        vectors_before,
        vectors_after,
    ) = {
        let src = p.db.inner.read();
        // Raw counts (everything, including `~valise.schema` doc frames) drive
        // the no-op decision; the user-facing report excludes doc frames.
        let reserved = schema_doc::schema_collection_id(&src);
        let is_user = |cid: CollectionId| Some(cid) != reserved;
        let raw_frames_before = src.frames().len();
        let raw_frames_after = src.active_frames().count();
        let frames_before = src
            .frames()
            .iter()
            .filter(|f| is_user(f.collection_id))
            .count();
        let frames_after = src
            .active_frames()
            .filter(|f| is_user(f.collection_id))
            .count();
        let vectors_before = src.vectors().len();
        let vectors_after = src
            .vectors()
            .iter()
            .filter(|v| v.status == VectorStatus::Active)
            .count();
        (
            src.path().to_path_buf(),
            raw_frames_before,
            raw_frames_after,
            frames_before,
            frames_after,
            vectors_before,
            vectors_after,
        )
    };
    let bytes_before = std::fs::metadata(&src_path).map(|m| m.len()).unwrap_or(0);

    // No-op fast path: nothing tombstoned → rewriting would only cost a QAM
    // round-trip for no reclaim. Decided on the raw counts so a tombstoned
    // schema doc (additive redeclare) is still reclaimable.
    if raw_frames_before == raw_frames_after && vectors_before == vectors_after {
        return Ok(CompactReport {
            compacted: false,
            frames_before,
            frames_after,
            vectors_before,
            vectors_after,
            bytes_before,
            bytes_after: bytes_before,
        });
    }

    // Build the fresh file at a temp path (clobber any stale one from a crashed
    // prior compact so `create_new` doesn't wedge).
    let temp_path = src_path.with_extension("valise.compacting");
    let _ = std::fs::remove_file(&temp_path);
    build_compacted_file(&p, &temp_path, opts)?;

    // Swap it in. The temp was committed+fsync'd; rename is atomic, and the
    // directory fsync makes the new name durable across power loss.
    std::fs::rename(&temp_path, &src_path)?;
    crate::io::sync_parent_dir(&src_path, Durability::FullSync)?;

    // Reopen and publish a fresh {db, schema, identity} triple atomically.
    // Collection shapes are restored from the persisted schema docs (which
    // streamed through verbatim as active frames) by `rebuild` — the old
    // in-process shape carry-forward is gone, which is what fixes the
    // latent compact-after-reopen text-indexing drop.
    let new_db = Database::open_read_write(&src_path)?;
    new_db.set_durability(Durability::Buffered);
    let mut new_schema = SchemaRegistry::default();
    let mut new_identity = IdentityIndex::default();
    {
        let valise = new_db.inner.read();
        new_schema.rebuild(&valise)?;
        new_identity.rebuild(&valise)?;
    }
    handles.published.store(Arc::new(Published {
        db: new_db,
        schema: Arc::new(Mutex::new(new_schema)),
        identity: Arc::new(Mutex::new(new_identity)),
    }));

    let bytes_after = std::fs::metadata(&src_path).map(|m| m.len()).unwrap_or(0);
    Ok(CompactReport {
        compacted: true,
        frames_before,
        frames_after,
        vectors_before,
        vectors_after,
        bytes_before,
        bytes_after,
    })
}

/// Stream the live set of `p` into a fresh committed Valise file at `temp_path`.
fn build_compacted_file(p: &Published, temp_path: &Path, opts: &CompactOptions) -> Result<()> {
    let mut dst = ValiseFile::create_with_options(temp_path, CreateOptions::default())?;
    let src = p.db.inner.read();

    // Per-collection text space (old id) — the frame→text-space binding —
    // derived from the PERSISTED schema docs in `src`, not from in-process
    // shapes. The in-process derivation silently dropped text indexing when
    // compacting a reopened store; the docs are always present in the file.
    let coll_text: HashMap<CollectionId, TextSpaceId> = {
        let coll_by_name: HashMap<&str, CollectionId> = src
            .collections()
            .iter()
            .map(|c| (c.name.as_str(), c.collection_id))
            .collect();
        let ts_by_name: HashMap<&str, TextSpaceId> = src
            .text_spaces()
            .iter()
            .map(|t| (t.name.as_str(), t.text_space_id))
            .collect();
        let mut m = HashMap::new();
        for (coll_name, doc) in schema_doc::load_all(&src)? {
            let Some(&cid) = coll_by_name.get(coll_name.as_str()) else {
                continue;
            };
            if let Some(field) = doc.fields.iter().find(|f| f.kind == schema_doc::KIND_TEXT)
                && let Some(&tsid) = ts_by_name.get(field.space.as_str())
            {
                m.insert(cid, tsid);
            }
        }
        m
    };

    // ---- schema: re-register everything verbatim, mapping old ids → new ----
    let mut analyzer_map = HashMap::new();
    for a in src.analyzers() {
        analyzer_map.insert(a.analyzer_id, dst.register_analyzer(a.clone())?);
    }
    let mut fs_map = HashMap::new();
    for f in src.field_schemas() {
        fs_map.insert(f.field_schema_id, dst.register_field_schema(f.clone())?);
    }
    let mut prof_map = HashMap::new();
    for pr in src.retrieval_profiles() {
        prof_map.insert(pr.profile_id, dst.register_retrieval_profile(pr.clone())?);
    }
    let mut text_map: HashMap<TextSpaceId, TextSpaceId> = HashMap::new();
    for ts in src.text_spaces() {
        let new = dst.register_text_space(TextSpaceDesc {
            text_space_id: TextSpaceId(0),
            name: ts.name.clone(),
            analyzer_id: analyzer_map[&ts.analyzer_id],
            field_schema_id: fs_map[&ts.field_schema_id],
            default_profile_id: prof_map[&ts.default_profile_id],
            flags: ts.flags,
            enabled_retrievers: ts.enabled_retrievers,
        })?;
        text_map.insert(ts.text_space_id, new);
    }
    let mut space_map: HashMap<EmbeddingSpaceId, EmbeddingSpaceId> = HashMap::new();
    for es in src.embedding_spaces() {
        let new_codec = match es.primary_codec_id {
            Some(cid) => {
                let params = src.codec_params(cid)?;
                let sample: Vec<Vec<f32>> = if opts.recalibrate {
                    src.active_vectors(es.embedding_space_id)
                        .take(RECALIBRATE_SAMPLE)
                        .map(|v| {
                            src.read_vector(v.vector_id, Reconstruct::F32Vector)
                                .map(|r| r.expect_f32())
                        })
                        .collect::<Result<_>>()?
                } else {
                    Vec::new()
                };
                if opts.recalibrate && !sample.is_empty() {
                    // Re-fit the same family at the same operating point.
                    Some(match &params {
                        CodecParams::QamLloydMax(p) => dst
                            .register_codec_qam_from_sample_with_bits(
                                es.dimension as usize,
                                p.amp_bits,
                                p.phase_bits,
                                &sample,
                            )?,
                        CodecParams::Upq(p) => dst.register_codec_upq_from_sample_with_options(
                            es.dimension as usize,
                            p.cells,
                            p.design_source,
                            &sample,
                        )?,
                    })
                } else {
                    // Reuse the existing params (near-idempotent re-encode).
                    Some(dst.register_codec(params)?)
                }
            }
            None => None,
        };
        let new = dst.register_embedding_space(EmbeddingSpaceSpec {
            provider: es.provider.clone(),
            model: es.model.clone(),
            dimension: es.dimension,
            metric: es.metric,
            normalized: es.normalized,
            dtype: es.dtype,
            primary_codec_id: new_codec,
            secondary_codec_id: None,
        })?;
        space_map.insert(es.embedding_space_id, new);
    }
    let mut coll_map: HashMap<CollectionId, CollectionId> = HashMap::new();
    for c in src.collections() {
        coll_map.insert(
            c.collection_id,
            dst.create_collection_at(c.name.clone(), c.created_at)?,
        );
    }

    // ---- frames: stream active in frame-id order (parents before children;
    // preserves the source's valid time ordering). Capture the needed fields
    // once from the borrowed `&FrameDesc` (free), so the rebuild loop avoids a
    // redundant `frame_full` clone per frame; a parent that was tombstoned
    // re-parents the child to root (`None`). ----
    struct FrameMeta {
        frame_id: FrameId,
        collection_id: CollectionId,
        role: FrameRole,
        created_at: i64,
        parent: Option<FrameId>,
        chunk_index: Option<u32>,
        chunk_count: Option<u32>,
        has_payload: bool,
        has_uri: bool,
    }
    let active: Vec<FrameMeta> = src
        .active_frames()
        .map(|f| FrameMeta {
            frame_id: f.frame_id,
            collection_id: f.collection_id,
            role: f.role,
            created_at: f.created_at,
            parent: f.parent_frame_id,
            chunk_index: f.chunk_index,
            chunk_count: f.chunk_count,
            has_payload: f.payload_ref.is_some(),
            has_uri: f.uri_ref.is_some(),
        })
        .collect();
    let mut frame_map: FxHashMap<FrameId, FrameId> =
        FxHashMap::with_capacity_and_hasher(active.len(), Default::default());
    for m in &active {
        let payload = if m.has_payload {
            src.read_payload(m.frame_id)?
        } else {
            Vec::new()
        };
        let uri = if m.has_uri {
            src.read_frame_uri(m.frame_id)?
        } else {
            None
        };
        let new_parent = m.parent.and_then(|op| frame_map.get(&op).copied());
        let new_fid = dst.put_frame(PutFrame {
            collection_id: coll_map[&m.collection_id],
            role: m.role,
            payload: &payload,
            created_at: Some(m.created_at),
            updated_at: None,
            parent_frame_id: new_parent,
            chunk_index: m.chunk_index,
            chunk_count: m.chunk_count,
            uri: uri.as_deref(),
        })?;
        frame_map.insert(m.frame_id, new_fid);
        if !payload.is_empty()
            && let Some(old_tsid) = coll_text.get(&m.collection_id)
        {
            dst.index_frame_text(new_fid, text_map[old_tsid])?;
        }
    }

    // ---- vectors: stream active per space, re-encode into the new codec ----
    let space_ids: Vec<EmbeddingSpaceId> = src
        .embedding_spaces()
        .iter()
        .map(|e| e.embedding_space_id)
        .collect();
    for old_space in space_ids {
        let new_space = space_map[&old_space];
        let owned: Vec<(FrameId, crate::format::VectorId)> = src
            .active_vectors(old_space)
            .map(|v| (v.owner_frame_id, v.vector_id))
            .collect();
        for (owner, vid) in owned {
            let Some(new_owner) = frame_map.get(&owner).copied() else {
                continue;
            };
            let values = src.read_vector(vid, Reconstruct::F32Vector)?.expect_f32();
            dst.put_vector(PutVector {
                owner_frame_id: new_owner,
                embedding_space_id: new_space,
                values: &values,
            })?;
        }
    }

    dst.commit()?;
    Ok(())
}
