//! Application-facing CRUD layer over the Valise engine.
//!
//! `Store` → `Collection` → `Record` with sane defaults, stable record
//! identity (so `put`/`delete`/`get` work by application key over an
//! append-only engine), a declarative [`Schema`] model with per-field codec
//! choice, and a single serialized writer with group-committed durability.
//! See `docs/DB_LAYER_PLAN.md` and `docs/SIMPLE_API_SPEC.md`.
//!
//! The full layer (DB-layer plan §13 P1–P4) is implemented:
//! - **P1 — schema + write:** [`Store`], [`Schema`] field specs
//!   ([`Text`]/[`Vector`], with [`Codec`]/[`Calibrate`] choices), shared
//!   spaces ([`Store::define_space`]), [`Record`], identity, the single
//!   serialized [`Writer`] (`put`/`delete`/`commit`), `get`, and deferred
//!   calibration.
//! - **P2 — read:** [`Reader`] (`get`/`get_many`/`search`), the [`Search`]
//!   builder ([`TextScorer`]/[`Fusion`]/[`Rerank`]/[`Recency`]), layer-side
//!   fusion, all write-free.
//! - **P3 — scale-out:** [`Partitioned`]/[`View`]/[`Window`] routing, the
//!   global-fusion `search_view`, soft `forget_before`, and the [`Bulk`] loader.
//! - **P4 — maintenance:** [`Store::compact`] (GC of tombstoned bytes via
//!   rewrite + atomic swap), [`Store::stats`] / [`StoreStats`], advisory
//!   compaction policy (nothing fires inside writes).
//!
//! Deferred to later work: crypto-shred hard-erase, f8 ingest, push-down tag
//! filtering, and an in-place vote-index rebuild.
//!
//! ```no_run
//! use valise::db::{Record, Schema, Search, Store, Vector};
//!
//! let store = Store::open("data.vls")?; // open-or-create
//! store.collection("knowledge", Schema::new()
//!     .text("body")                     // English BM25 (default)
//!     .vector("dense", Vector::dim(768)))?; // cosine, auto-calibrating QAM
//!
//! let mut w = store.writer();
//! let emb = vec![0.0f32; 768];
//! w.put("knowledge", "doc-1",
//!     Record::new().text("body", "vector quantization").vector("dense", &emb))?;
//! w.commit()?;
//!
//! let hits = store.search("knowledge", Search::new()
//!     .text("body", "quantization")
//!     .vector("dense", &emb)
//!     .top_k(10))?;
//! # Ok::<(), valise::Error>(())
//! ```

mod collection;
mod compact;
mod identity;
mod loader;
mod partition;
mod query;
mod reader;
mod record;
mod schema;
mod schema_doc;
mod space;
mod writer;

use std::path::Path;
use std::sync::Arc;

use arc_swap::ArcSwap;
use parking_lot::Mutex;

use crate::concurrency::database::Database;
use crate::error::Result;

use self::identity::IdentityIndex;
use self::space::{CollectionRecord, SchemaRegistry, check_redeclare, reject_reserved_name};

pub use self::collection::{Collection, CollectionInfo};
pub use self::compact::{CompactOptions, CompactReport};
pub use self::loader::Bulk;
pub use self::partition::{Partition, Partitioned, View, Window};
pub use self::query::{Fusion, Hit, Recency, Rerank, Search, SearchResult, TextScorer, TfMode};
pub use self::reader::Reader;
pub use self::record::{Key, Record, Stored, Value};
pub use self::schema::{Calibrate, Codec, FieldSpec, Schema, Text, UpqDesign, Vector};
pub use self::space::{Lang, Metric, Space, SpaceInfo, SpaceKind, SpaceRef};
pub use self::writer::{OwnedWriter, Writer};
pub use crate::io::Durability;

/// Options for opening or creating a [`Store`].
#[derive(Clone, Debug)]
pub struct StoreOptions {
    /// Per-call durability. Defaults to [`Durability::Buffered`] so the
    /// group-committed write path actually amortizes fsync (DB-layer plan §8);
    /// `commit()` is the durability barrier.
    pub durability: Durability,
}

impl Default for StoreOptions {
    fn default() -> Self {
        Self {
            durability: Durability::Buffered,
        }
    }
}

/// The atomically-swappable engine state: the engine handle plus the schema
/// and identity indexes that are all consistent with it. Compaction builds a
/// fresh `Published` (new file, rebuilt indexes) and swaps it in with a single
/// pointer store, so a reader that loads it observes a self-consistent triple
/// (no torn db/identity).
pub(crate) struct Published {
    pub(crate) db: Arc<Database>,
    pub(crate) schema: Arc<Mutex<SchemaRegistry>>,
    pub(crate) identity: Arc<Mutex<IdentityIndex>>,
}

/// The shared, cloneable core every handle hangs off. `published` is the
/// swap slot; `writer_lock` is the (un-swapped) single-writer mutex so writers
/// and compaction serialize on the same lock across a swap.
#[derive(Clone)]
pub(crate) struct Handles {
    pub(crate) published: Arc<ArcSwap<Published>>,
    /// Enforces single-writer. `parking_lot::Mutex` is **not reentrant** — do
    /// not acquire a second writer (e.g. `Partitioned::forget_before`) while one
    /// is already held on the same thread.
    pub(crate) writer_lock: Arc<Mutex<()>>,
}

impl Handles {
    /// Load the current published state.
    pub(crate) fn current(&self) -> Arc<Published> {
        self.published.load_full()
    }

    /// Acquire the single serialized writer (blocks until any prior one drops).
    /// The writer captures the current `Published` for its lifetime; the writer
    /// lock excludes compaction, so the captured state can't be swapped under it.
    pub(crate) fn writer(&self) -> Writer<'_> {
        let guard = self.writer_lock.lock();
        let p = self.published.load_full();
        Writer::new(
            guard,
            Arc::clone(&p.db),
            Arc::clone(&p.schema),
            Arc::clone(&p.identity),
        )
    }

    /// Acquire an owned single serialized writer (no borrow of the store).
    /// Mirrors [`Handles::writer`] but takes the same writer lock by `Arc`, so
    /// the returned guard carries no lifetime. Contends on the same underlying
    /// `RawMutex`, preserving the single-writer invariant.
    pub(crate) fn writer_owned(&self) -> OwnedWriter {
        let guard = self.writer_lock.lock_arc();
        let p = self.published.load_full();
        OwnedWriter::new(
            guard,
            Arc::clone(&p.db),
            Arc::clone(&p.schema),
            Arc::clone(&p.identity),
        )
    }

    /// Non-blocking variant of [`Handles::writer_owned`]: returns `None` if the
    /// single-writer lock is already held. Lets a caller (e.g. an FFI thread)
    /// poll for the lock without parking, so it can release the GIL between
    /// attempts and avoid an interpreter-wide deadlock.
    pub(crate) fn try_writer_owned(&self) -> Option<OwnedWriter> {
        let guard = self.writer_lock.try_lock_arc()?;
        let p = self.published.load_full();
        Some(OwnedWriter::new(
            guard,
            Arc::clone(&p.db),
            Arc::clone(&p.schema),
            Arc::clone(&p.identity),
        ))
    }

    /// A cheap live read handle. Reloads the published state per operation.
    pub(crate) fn reader(&self) -> Reader {
        Reader::new(Arc::clone(&self.published))
    }
}

/// The application-facing store: a handle over the engine plus the three things
/// the engine doesn't provide — identity, schema-by-name, and (P4) compaction.
pub struct Store {
    handles: Handles,
}

impl Store {
    /// Open an existing store or create one if the path does not exist
    /// (default options: [`Durability::Buffered`], `commit()` is the
    /// durability barrier).
    pub fn open(path: impl AsRef<Path>) -> Result<Store> {
        Self::open_with(path, StoreOptions::default())
    }

    /// [`Store::open`] with explicit [`StoreOptions`].
    pub fn open_with(path: impl AsRef<Path>, opts: StoreOptions) -> Result<Store> {
        let path = path.as_ref();
        if path.exists() {
            let db = Database::open_read_write(path)?;
            Self::from_db(db, opts, /* rebuild = */ true)
        } else {
            let db = Database::create(path)?;
            Self::from_db(db, opts, /* rebuild = */ false)
        }
    }

    /// Create a new store, failing if the path already exists.
    pub fn create(path: impl AsRef<Path>) -> Result<Store> {
        Self::create_with(path, StoreOptions::default())
    }

    /// [`Store::create`] with explicit [`StoreOptions`].
    pub fn create_with(path: impl AsRef<Path>, opts: StoreOptions) -> Result<Store> {
        let db = Database::create(path)?;
        Self::from_db(db, opts, /* rebuild = */ false)
    }

    fn from_db(db: Arc<Database>, opts: StoreOptions, rebuild: bool) -> Result<Store> {
        db.set_durability(opts.durability);
        let mut schema = SchemaRegistry::default();
        let mut identity = IdentityIndex::default();
        if rebuild {
            let valise = db.inner.read();
            schema.rebuild(&valise)?;
            identity.rebuild(&valise)?;
        }
        Ok(Store {
            handles: Handles {
                published: Arc::new(ArcSwap::from_pointee(Published {
                    db,
                    schema: Arc::new(Mutex::new(schema)),
                    identity: Arc::new(Mutex::new(identity)),
                })),
                writer_lock: Arc::new(Mutex::new(())),
            },
        })
    }

    // ---- Schema (idempotent by name) ----

    /// Define (or return the existing) shared space from a field spec —
    /// `Text::english()` / `Vector::dim(768).codec(Codec::upq())` / … —
    /// then bind it in any schema with [`Text::space`] / [`Vector::space`].
    ///
    /// Vector specs with [`Calibrate::now`] or [`Codec::from_params`]
    /// register their codec eagerly; [`Calibrate::auto`] defers to the first
    /// commit containing vectors. Re-defining an existing name is a no-op
    /// for an identical spec and an error for a divergent one.
    pub fn define_space(&self, name: &str, spec: impl Into<FieldSpec>) -> Result<Space> {
        reject_reserved_name("define_space: space", name)?;
        let spec = spec.into();
        let p = self.handles.current();
        let mut schema = p.schema.lock();
        let mut valise = p.db.inner.write();
        schema.define_space(&mut valise, name, &spec)
    }

    /// Look up a previously defined space by name.
    #[must_use]
    pub fn space(&self, name: &str) -> Option<Space> {
        let p = self.handles.current();
        let schema = p.schema.lock();
        if let Some(e) = schema.vector_spaces.get(name) {
            Some(Space::vector(name, e.dim))
        } else if schema.text_spaces.contains_key(name) {
            Some(Space::text(name))
        } else {
            None
        }
    }

    /// All defined spaces — shared and `~auto/{collection}/{field}` ones
    /// (the latter flagged `auto`) — sorted by name.
    #[must_use]
    pub fn spaces(&self) -> Vec<SpaceInfo> {
        let p = self.handles.current();
        let schema = p.schema.lock();
        schema.space_infos()
    }

    // ---- Collections ----

    /// Create-or-open a collection, bind `schema` to it, and persist the
    /// schema in-file — so a later [`Store::open`] restores it and reopen
    /// needs **no re-declaration**.
    ///
    /// Inline field specs auto-define one private space per field (named
    /// `~auto/{name}/{field}`); shared bindings reuse spaces from
    /// [`Store::define_space`]. Re-declaring is a no-op when identical
    /// (canonical ordered comparison against the persisted schema) and
    /// accepted when purely additive (new fields appended, prior fields
    /// unchanged — the persisted doc is rewritten); anything else is
    /// `Error::SchemaMismatch` with field-level detail.
    ///
    /// Durability is **stage-only**: the persisted schema rides the next
    /// commit, exactly like the collection creation itself. A declare with
    /// no following commit leaves nothing in the file.
    ///
    /// Files written before schemas were persisted come back *shape-less*
    /// (reads/writes error with "has no schema"); one re-declaration via
    /// this method persists the schema, and every later open just works.
    pub fn collection(&self, name: &str, schema: Schema) -> Result<Collection> {
        reject_reserved_name("collection: collection", name)?;
        let p = self.handles.current();
        let mut reg = p.schema.lock();
        // Compare against the in-process doc (covers staged, pre-commit
        // declares whose frame payloads plan_sync cannot read yet) and the
        // persisted doc BEFORE binding, so a divergent re-declaration errors
        // with zero side effects (no spaces defined, nothing staged).
        let doc = schema_doc::SchemaDoc::from_schema(name, &schema)?;
        if let Some(rec) = reg.collections.get(name) {
            if let Some(prev) = &rec.doc {
                schema_doc::check_compatible(name, prev, &doc)?;
            } else if let Some(existing) = &rec.shape {
                check_redeclare(name, existing, &schema)?;
            }
        }
        let plan = {
            let valise = p.db.inner.read();
            schema_doc::plan_sync(&valise, name, &doc)?
        };
        let shape = {
            let mut valise = p.db.inner.write();
            reg.bind_schema(&mut valise, name, &schema)?
        };
        let collection_id = ensure_collection(&p.db, &mut reg, name)?;
        {
            let mut valise = p.db.inner.write();
            schema_doc::apply_sync(&mut valise, name, &doc, &plan)?;
        }
        reg.collections.insert(
            name.to_owned(),
            CollectionRecord {
                collection_id,
                shape: Some(shape),
                doc: Some(doc),
            },
        );
        Ok(Collection {
            id: collection_id,
            name: name.to_owned(),
        })
    }

    /// All user collections known to this store (engine + this process's
    /// schema). Layer-internal collections (the hidden `~valise.schema` doc
    /// store) are excluded.
    #[must_use]
    pub fn collections(&self) -> Vec<CollectionInfo> {
        let p = self.handles.current();
        let schema = p.schema.lock();
        schema
            .collections
            .iter()
            .map(|(name, rec)| CollectionInfo {
                name: name.clone(),
                id: rec.collection_id,
            })
            .collect()
    }

    /// Create-or-open a logical partitioned collection: records routed by
    /// `by` into physical collections (`"{base}:{period}"`) that share one
    /// resolved schema (inline specs auto-define `~auto/{base}/{field}`
    /// spaces once — no per-partition codec duplication).
    pub fn partitioned(&self, base: &str, schema: Schema, by: Partition) -> Result<Partitioned> {
        reject_reserved_name("partitioned: collection", base)?;
        let p = self.handles.current();
        let mut reg = p.schema.lock();
        let shape = {
            let mut valise = p.db.inner.write();
            reg.bind_schema(&mut valise, base, &schema)?
        };
        // The declared schema travels with the handle so lazily created
        // partition collections can persist their own schema docs (sharing
        // the base's `~auto/{base}/{field}` spaces).
        Ok(Partitioned::new(
            self.handles.clone(),
            base,
            schema,
            shape,
            by,
        ))
    }

    // ---- Handles ----

    /// Acquire the single serialized writer. Blocks until any prior writer is
    /// dropped.
    #[must_use]
    pub fn writer(&self) -> Writer<'_> {
        self.handles.writer()
    }

    /// Acquire an owned single serialized writer (no borrow of the `Store`).
    /// Blocks until any prior writer (owned or borrowed) drops; drop the
    /// returned [`OwnedWriter`] to release the single-writer lock. Useful where
    /// a writer handle must outlive a borrow of the `Store` (e.g. FFI objects).
    #[must_use]
    pub fn writer_owned(&self) -> OwnedWriter {
        self.handles.writer_owned()
    }

    /// Non-blocking [`Store::writer_owned`]: returns `None` if another writer
    /// currently holds the single-writer lock. The caller can poll this and
    /// yield between attempts (e.g. an FFI thread releasing the GIL while it
    /// waits) instead of parking the calling thread.
    #[must_use]
    pub fn try_writer_owned(&self) -> Option<OwnedWriter> {
        self.handles.try_writer_owned()
    }

    /// Acquire a read handle (cheap, `Clone + Send`, last-committed-wins).
    #[must_use]
    pub fn reader(&self) -> Reader {
        self.handles.reader()
    }

    // ---- Read ----

    /// Fetch a record by key. Convenience that delegates to [`Reader::get`].
    pub fn get(&self, coll: &str, key: impl Into<Key>) -> Result<Option<Stored>> {
        self.reader().get(coll, key)
    }

    /// Run a search. Convenience that delegates to [`Reader::search`].
    pub fn search(&self, coll: &str, q: Search) -> Result<SearchResult> {
        self.reader().search(coll, q)
    }

    /// Search across a partitioned view. Delegates to [`Reader::search_view`].
    pub fn search_view(&self, view: &View, q: Search) -> Result<SearchResult> {
        self.reader().search_view(view, q)
    }

    /// Escape hatch to the underlying engine handle (the current one; a
    /// compaction swaps in a new handle).
    #[must_use]
    pub fn raw(&self) -> Arc<Database> {
        Arc::clone(&self.handles.current().db)
    }

    // ---- Maintenance ----

    /// Reclaim tombstoned bytes (from `delete` / `forget_before`) by rewriting
    /// the live set into a fresh file and atomically swapping it in. No-op if
    /// nothing is tombstoned. Acquires the single writer for the whole op — do
    /// **not** call while holding a [`Writer`] on the same thread.
    pub fn compact(&self, opts: CompactOptions) -> Result<CompactReport> {
        compact::compact(&self.handles, &opts)
    }

    /// Live/total counts and tombstone ratio. Use it to drive compaction
    /// policy (`needs_compaction`); compaction is always explicit — nothing
    /// fires automatically inside writes. Counts cover user data only: the
    /// hidden `~valise.schema` doc frames are excluded.
    #[must_use]
    pub fn stats(&self) -> StoreStats {
        let p = self.handles.current();
        let valise = p.db.inner.read();
        let reserved = schema_doc::schema_collection_id(&valise);
        let is_user = |cid: crate::format::CollectionId| Some(cid) != reserved;
        let frames_total = valise
            .frames()
            .iter()
            .filter(|f| is_user(f.collection_id))
            .count();
        let frames_active = valise
            .active_frames()
            .filter(|f| is_user(f.collection_id))
            .count();
        let vectors_total = valise.vectors().len();
        let vectors_active = valise
            .vectors()
            .iter()
            .filter(|v| v.status == crate::format::catalog::VectorStatus::Active)
            .count();
        let file_bytes = std::fs::metadata(valise.path())
            .map(|m| m.len())
            .unwrap_or(0);
        StoreStats {
            frames_total,
            frames_active,
            vectors_total,
            vectors_active,
            file_bytes,
        }
    }
}

/// Snapshot of store occupancy. Drives compaction policy.
#[derive(Clone, Copy, Debug)]
pub struct StoreStats {
    pub frames_total: usize,
    pub frames_active: usize,
    pub vectors_total: usize,
    pub vectors_active: usize,
    pub file_bytes: u64,
}

impl StoreStats {
    /// Fraction of frames that are tombstoned, in `[0, 1]`.
    #[must_use]
    pub fn tombstone_ratio(&self) -> f32 {
        if self.frames_total == 0 {
            0.0
        } else {
            (self.frames_total - self.frames_active) as f32 / self.frames_total as f32
        }
    }

    /// Whether the tombstone ratio meets/exceeds `threshold` (e.g. `0.3`).
    #[must_use]
    pub fn needs_compaction(&self, threshold: f32) -> bool {
        self.tombstone_ratio() >= threshold
    }
}

/// Create-or-open a collection by name under an already-held schema lock,
/// returning its id. Shared by `Store::collection` and the partition router so
/// lazy partition creation goes through one schema-guarded path.
pub(crate) fn ensure_collection(
    db: &Arc<Database>,
    schema: &mut SchemaRegistry,
    name: &str,
) -> Result<crate::format::CollectionId> {
    if let Some(rec) = schema.collections.get(name) {
        return Ok(rec.collection_id);
    }
    let id = db.inner.write().create_collection(name)?;
    Ok(id)
}
