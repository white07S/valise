//! `Database` — registry-managed shared handle to an Valise file.
//!
//! Stage 4 deliverable. Multiple `Arc<Database>` clones in one process
//! all point at the same underlying `ValiseFile`; the global registry keyed
//! by `(dev, ino)` ensures `Database::open(path)` is idempotent within a
//! process — opening the same file twice (by absolute path, by relative
//! path, by symlink) collapses to one shared handle.
//!
//! See `docs/CONCURRENCY_PLAN.md` §6.
//!
//! ## Internal structure
//!
//! `Database` holds the underlying `ValiseFile` behind a `parking_lot::RwLock`.
//! Read paths take the read guard briefly per call (multi-reader
//! concurrent); write paths take the write guard for the whole
//! `WriteConnection` lifetime (single-writer-per-process).
//!
//! Stage 5's group-commit pipeline will replace the write-side `RwLock`
//! semantics with a queueing mutex so writers serialize their commits
//! without blocking readers; for Stage 4 the `RwLock` shape is the
//! simplest correct implementation.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex, Weak};

use parking_lot::RwLock;

use crate::concurrency::connection::{ReadConnection, WriteConnection};
use crate::concurrency::snapshot::Snapshot;
use crate::concurrency::writer_pipeline::WriterPipeline;
use crate::error::{Error, Result};
use crate::file::{CreateOptions, OpenMode, ValiseFile};

/// Global per-`(dev, ino)` registry. `Weak` so a `Database` whose last
/// `Arc` clone drops automatically removes itself.
static DATABASE_REGISTRY: LazyLock<Mutex<HashMap<(u64, u64), Weak<Database>>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Identity for a file in the registry: `(st_dev, st_ino)`. Stable
/// across rename, hard-link, and symlink. The kernel guarantees this
/// pair uniquely identifies a file within a single mount.
type InodeKey = (u64, u64);

/// Per-`(dev, ino)` shared handle. See module docs.
pub struct Database {
    /// The underlying `ValiseFile`, behind a `RwLock`. Stage 5: write
    /// methods take the write guard briefly per call, not for the
    /// connection's lifetime — cross-`WriteConnection` exclusion lives
    /// on the `WriterPipeline` instead. Reader paths take the read
    /// lock briefly per call.
    pub(crate) inner: RwLock<ValiseFile>,
    /// Group-commit pipeline. Provides single-writer-per-`Database`
    /// exclusion via its own mutex (decoupled from `inner`'s `RwLock`,
    /// so long-lived writers no longer block readers — Stage 4
    /// regression resolved). Stage 5 follow-up will route concurrent
    /// commits through the leader/followers protocol for fsync
    /// coalescing.
    pub(crate) pipeline: WriterPipeline,
    /// `(dev, ino)` identifying this file in the registry. Used at
    /// `Drop` to evict the registry entry.
    inode_key: InodeKey,
}

impl Database {
    /// Open an Valise file in `ReadWrite` mode. Returns an `Arc<Database>`
    /// that may be shared with `clone()` across threads.
    pub fn open_read_write(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        Self::open(path, OpenMode::ReadWrite)
    }

    /// Open an Valise file in `ReadOnly` mode.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        Self::open(path, OpenMode::ReadOnly)
    }

    /// Open an existing Valise file. The mode dispatches to the right
    /// `ValiseFile` constructor; the registry collapses repeat opens of
    /// the same `(dev, ino)` to a single shared handle within the process.
    ///
    /// **Mode reconciliation**: if a registered handle already exists for
    /// this file, we return that handle regardless of the requested
    /// mode. This matches single-writer-per-process semantics — the
    /// caller cannot upgrade or downgrade an existing shared handle.
    /// Document this at the call site.
    pub fn open(path: impl AsRef<Path>, mode: OpenMode) -> Result<Arc<Self>> {
        let path = path.as_ref();
        let inode_key = inode_for_path(path)?;

        // Fast path: registry hit.
        if let Some(existing) = lookup_registered(inode_key) {
            return Ok(existing);
        }

        // Open the underlying file. Drops out of the slow path on error
        // so the registry doesn't get a dangling entry.
        let valise = ValiseFile::open(path, mode)?;

        // Slow-path race: another thread may have opened-and-registered
        // for the same `(dev, ino)` between our `lookup_registered` and
        // here. Resolve under the registry lock; if a winner exists we
        // discard our `valise` and return theirs.
        let mut registry = DATABASE_REGISTRY
            .lock()
            .map_err(|_| Error::Other("database registry poisoned".into()))?;
        if let Some(weak) = registry.get(&inode_key)
            && let Some(arc) = weak.upgrade()
        {
            // Someone else won — return theirs and drop ours.
            return Ok(arc);
        }
        let pipeline = WriterPipeline::new();
        // Stage 5++: route the underlying `ValiseFile`'s per-commit fsync
        // through the pipeline's `GroupFsync` barrier so concurrent
        // committers share one F_FULLFSYNC.
        valise.install_commit_fsync_barrier(Arc::clone(&pipeline.commit_fsync));
        let db = Arc::new(Database {
            inner: RwLock::new(valise),
            pipeline,
            inode_key,
        });
        registry.insert(inode_key, Arc::downgrade(&db));
        Ok(db)
    }

    /// Create a new Valise file. Always inserts a fresh entry in the
    /// registry (a new file has a new inode by construction).
    pub fn create(path: impl AsRef<Path>) -> Result<Arc<Self>> {
        Self::create_with_options(path, CreateOptions::default())
    }

    /// Create a new Valise file with non-default options.
    pub fn create_with_options(
        path: impl AsRef<Path>,
        options: CreateOptions,
    ) -> Result<Arc<Self>> {
        let path = path.as_ref();
        let valise = ValiseFile::create_with_options(path, options)?;
        let inode_key = inode_for_path(path)?;
        let mut registry = DATABASE_REGISTRY
            .lock()
            .map_err(|_| Error::Other("database registry poisoned".into()))?;
        // A `create` returning `Ok` implies the path was new (O_CREATE_NEW
        // semantics inside `ValiseFile::create_with_options`), so a stale
        // registry entry would be a bug — but we tolerate it by falling
        // back to inserting our fresh handle.
        let pipeline = WriterPipeline::new();
        valise.install_commit_fsync_barrier(Arc::clone(&pipeline.commit_fsync));
        let db = Arc::new(Database {
            inner: RwLock::new(valise),
            pipeline,
            inode_key,
        });
        registry.insert(inode_key, Arc::downgrade(&db));
        Ok(db)
    }

    /// Atomic load of the most recently published `Snapshot`. Cheap —
    /// one `ArcSwap::load_full`. Identical semantics to
    /// `ValiseFile::snapshot`.
    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.inner.read().snapshot()
    }

    /// Acquire a read connection. Pins an `Arc<Snapshot>` at
    /// acquisition; subsequent reads through this connection see a
    /// consistent view even across concurrent commits in the same
    /// process. Cheap: one snapshot `Arc::clone`.
    pub fn reader(self: &Arc<Self>) -> ReadConnection {
        ReadConnection::new(Arc::clone(self))
    }

    /// Acquire a write connection. This does **not** establish exclusive
    /// write access for the connection's lifetime: multiple
    /// `WriteConnection`s can coexist on one `Database` and share its
    /// in-memory mutation buffer and dirty set — a `commit()` from any one
    /// of them flushes everyone's pending writes (see [`WriteConnection`]
    /// for the shared-buffer semantics). Each write *method call* briefly
    /// takes the inner `RwLock` write guard and releases it; `commit()`
    /// additionally takes a cross-process exclusive byte lock for the
    /// duration of its disk writes. If you need isolated logical
    /// transactions, serialize writers at the application layer.
    pub fn writer(self: &Arc<Self>) -> WriteConnection {
        WriteConnection::new(Arc::clone(self))
    }

    /// `(dev, ino)` identifying this file. Useful for debugging and
    /// for tests that want to assert two `Database` handles point at
    /// the same underlying file.
    pub fn inode_key(&self) -> (u64, u64) {
        self.inode_key
    }

    /// Override the per-call durability mode for write operations on
    /// this `Database`. Defaults to `FullSync` (every `put_frame`/
    /// `put_vector`/etc. issues `F_FULLFSYNC`). For Stage 5++ avalanche
    /// commits to actually amortize: set this to `Buffered` so only
    /// `commit()` fsyncs (via the pipeline's `GroupFsync` barrier),
    /// and the per-call writes stay in the page cache.
    ///
    /// **Crash semantics**: under `Buffered`, only the prefix of writes
    /// covered by the most recent `commit()` is recoverable. Use this
    /// when you commit aggressively; do not use it for ad-hoc puts
    /// without commits.
    pub fn set_durability(&self, durability: crate::io::Durability) {
        self.inner.write().set_durability(durability);
    }
}

impl Drop for Database {
    fn drop(&mut self) {
        // Evict our entry from the registry. Use `if let` to tolerate a
        // poisoned mutex — a panic at this point would mask the real
        // failure mode.
        if let Ok(mut registry) = DATABASE_REGISTRY.lock() {
            // Only remove if the entry still points at us (a different
            // `Database` may have been inserted under the same
            // `(dev, ino)` between our last `Arc` decrement and the
            // `Drop` running, in unusual races).
            if let Some(weak) = registry.get(&self.inode_key)
                && weak.strong_count() == 0
            {
                registry.remove(&self.inode_key);
            }
        }
    }
}

/// Get the `(dev, ino)` for a path.
///
/// We use path-based `metadata` (not `fstat` on an open fd) because the
/// file we care about is opened *inside* `ValiseFile::open` and we need the
/// inode key *before* we open it (to short-circuit the registry hit
/// path). The TOCTOU window between this call and `ValiseFile::open` is
/// vanishingly small in practice; if a rename slips in, the registry
/// collapse may miss for that one call but correctness is preserved
/// (the handle returned is still a valid handle to the file the caller
/// asked for, just not registry-deduped against an existing handle for
/// the post-rename inode).
fn inode_for_path(path: &Path) -> Result<InodeKey> {
    use std::os::unix::fs::MetadataExt;
    let metadata = std::fs::metadata(path)?;
    Ok((metadata.dev(), metadata.ino()))
}

/// Look up an existing registered `Database` for the given inode key.
/// Returns `None` if no entry or the entry's `Weak` has expired.
fn lookup_registered(inode_key: InodeKey) -> Option<Arc<Database>> {
    let registry = DATABASE_REGISTRY.lock().ok()?;
    registry.get(&inode_key)?.upgrade()
}
