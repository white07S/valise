//! Reader and writer handles over a shared `Database`.
//!
//! Stage 4 deliverable. Two flavors:
//!
//! - [`ReadConnection`]: cheap, multiple per `Database`. Pins an
//!   `Arc<Snapshot>` at acquisition; reads through the connection see
//!   a consistent view across concurrent writes. Drop releases the
//!   pin (and may release an old mmap if no other snapshot holds it).
//!
//! - [`WriteConnection`]: at most one outstanding per `Database`. Holds
//!   the underlying writer lock for the connection's lifetime. Mutating
//!   methods (`put_frame`, `put_vector`, `commit`, …) live on the
//!   connection so the borrow checker enforces single-writer semantics
//!   end-to-end.
//!
//! See `docs/CONCURRENCY_PLAN.md` §6.

use std::path::PathBuf;
use std::sync::Arc;

use crate::concurrency::database::Database;
use crate::concurrency::snapshot::Snapshot;
use crate::error::Result;
use crate::file::{
    CommitOutcome, HybridHit, HybridQuery, PutFrame, PutVector, TextQuery, TimeQuery, VectorHit,
};
use crate::format::{
    CollectionId, FrameId, VectorId,
    catalog::{
        AnalyzerDesc, CodecDesc, CollectionDesc, EmbeddingSpaceDesc, FieldSchemaDesc, FrameDesc,
        FrameStub, FusionProfileDesc, RetrievalProfileDesc, TextSpaceDesc, VectorDesc,
    },
    header::Header,
};
use crate::retrieval::Hit;

/// Reader handle. Holds `Arc<Database>` (keeps the file open) and
/// `Arc<Snapshot>` (pins the view at acquisition time).
///
/// Many `ReadConnection`s may exist simultaneously; they coexist with
/// at most one `WriteConnection` per `Database`. Snapshot pin survives
/// concurrent commits — a long-running query sees the bytes in place at
/// acquisition even if the writer rotates the current mmap underneath.
///
/// Most read methods on `ReadConnection` defer to the inner `ValiseFile`
/// via the `Database`'s read-side `RwLock`. Methods that benefit from
/// the snapshot pin (multi-step queries, streaming) can call
/// [`Self::snapshot`] and use the snapshot directly.
pub struct ReadConnection {
    db: Arc<Database>,
    snapshot: Arc<Snapshot>,
}

impl ReadConnection {
    pub(crate) fn new(db: Arc<Database>) -> Self {
        let snapshot = db.snapshot();
        Self { db, snapshot }
    }

    /// The `Snapshot` pinned at this connection's acquisition. Stable
    /// across the connection's lifetime; call [`Self::refresh_snapshot`]
    /// to re-pin to the latest published snapshot.
    #[must_use]
    pub fn snapshot(&self) -> &Arc<Snapshot> {
        &self.snapshot
    }

    /// Re-pin to the latest published snapshot. Useful for long-lived
    /// connections that periodically want to see fresh data.
    pub fn refresh_snapshot(&mut self) {
        self.snapshot = self.db.snapshot();
    }

    /// The owning `Database`. Cheap clone of the inner `Arc`.
    #[must_use]
    pub fn database(&self) -> &Arc<Database> {
        &self.db
    }

    // ---- delegated read methods ----
    //
    // Each delegates to the underlying `ValiseFile` via the read-side
    // `RwLock`. Stage 5 will rewire these to drive off
    // `self.snapshot` directly, eliminating the per-call lock. For
    // Stage 4 the lock is brief and uncontended on the read path.

    pub fn path(&self) -> PathBuf {
        self.db.inner.read().path().to_path_buf()
    }

    pub fn header(&self) -> Header {
        self.db.inner.read().header().clone()
    }

    pub fn collections(&self) -> Vec<CollectionDesc> {
        self.db.inner.read().collections().to_vec()
    }

    pub fn frames(&self) -> Vec<FrameDesc> {
        self.db.inner.read().frames().to_vec()
    }

    pub fn frame_stubs(&self) -> Vec<FrameStub> {
        self.db.inner.read().frame_stubs().to_vec()
    }

    /// Resolve a frame to its full `FrameDesc` via the pinned snapshot.
    /// **Does not lock anything** — drives directly off
    /// `Snapshot::frame_full`. Mixed-mode reads (concurrent with a
    /// committing writer) no longer wait on the writer's commit window.
    pub fn frame_full(&self, frame_id: FrameId) -> Result<FrameDesc> {
        self.snapshot.frame_full(frame_id)
    }

    pub fn vectors(&self) -> Vec<VectorDesc> {
        self.db.inner.read().vectors().to_vec()
    }

    pub fn embedding_spaces(&self) -> Vec<EmbeddingSpaceDesc> {
        self.db.inner.read().embedding_spaces().to_vec()
    }

    pub fn codecs(&self) -> Vec<CodecDesc> {
        self.db.inner.read().codecs().to_vec()
    }

    pub fn text_spaces(&self) -> Vec<TextSpaceDesc> {
        self.db.inner.read().text_spaces().to_vec()
    }

    pub fn analyzers(&self) -> Vec<AnalyzerDesc> {
        self.db.inner.read().analyzers().to_vec()
    }

    pub fn field_schemas(&self) -> Vec<FieldSchemaDesc> {
        self.db.inner.read().field_schemas().to_vec()
    }

    pub fn retrieval_profiles(&self) -> Vec<RetrievalProfileDesc> {
        self.db.inner.read().retrieval_profiles().to_vec()
    }

    pub fn fusion_profiles(&self) -> Vec<FusionProfileDesc> {
        self.db.inner.read().fusion_profiles().to_vec()
    }

    /// Read a frame's payload bytes via the pinned snapshot. Lock-free
    /// — see [`Self::frame_full`].
    pub fn read_payload(&self, frame_id: FrameId) -> Result<Vec<u8>> {
        self.snapshot.read_payload(frame_id)
    }

    /// Read a frame's payload as UTF-8 text via the pinned snapshot.
    /// Lock-free.
    pub fn read_raw_text(&self, frame_id: FrameId) -> Result<String> {
        self.snapshot.read_raw_text(frame_id)
    }

    pub fn read_vector(
        &self,
        vector_id: VectorId,
        mode: crate::file::Reconstruct,
    ) -> Result<crate::file::ReadVectorResult> {
        self.db.inner.read().read_vector(vector_id, mode)
    }

    pub fn vector_search(&self, query: crate::file::VectorSearchQuery) -> Result<Vec<VectorHit>> {
        self.db.inner.read().vector_search(query)
    }

    pub fn query_text(&self, query: TextQuery) -> Result<Vec<Hit>> {
        self.db.inner.read().query_text(query)
    }

    pub fn query_hybrid(&self, query: HybridQuery) -> Result<Vec<HybridHit>> {
        self.db.inner.read().query_hybrid(query)
    }

    pub fn time_range_query(&self, query: TimeQuery) -> Vec<FrameId> {
        self.db.inner.read().time_range_query(query)
    }

    pub fn collection_member_frames(&self, collection_id: CollectionId) -> Vec<FrameId> {
        self.db
            .inner
            .read()
            .collection_member_frames(collection_id)
            .to_vec()
    }

    pub fn collection_member_vectors(&self, collection_id: CollectionId) -> Vec<VectorId> {
        self.db
            .inner
            .read()
            .collection_member_vectors(collection_id)
            .to_vec()
    }
}

/// Writer handle. Stage 5++ avalanche model:
///
/// - **Multiple `WriteConnection`s coexist on a single `Database`.** The
///   per-connection writer lock from Stage 5 is gone; the
///   `WriterPipeline`'s `commit_fsync` `GroupFsync` barrier is what
///   orchestrates concurrent committers now.
/// - Each write *method call* briefly takes the underlying `RwLock`
///   write guard, mutates `ValiseFile`, releases. Readers can interleave
///   freely between calls; concurrent `WriteConnection`s serialize on
///   that per-call write lock for the actual mutation, not for the
///   connection's lifetime.
/// - **Cross-connection puts share state.** Two `WriteConnection`s'
///   `put_frame` calls accumulate into the same in-memory catalog and
///   shared dirty set. A `commit()` from either flushes everyone's
///   pending writes — shared-buffer semantics for an embedded
///   multi-writer DB (there is no WAL; the v2 format removed it). If you
///   need isolated logical transactions, serialize at the application
///   layer.
/// - `commit()` runs as a *two-phase avalanche*: phase A (writes,
///   under the inner `RwLock` write guard) → drop the lock → phase
///   B's GroupFsync barrier (no lock; multiple committers pile in
///   here while the leader is parked on the ~10 ms `F_FULLFSYNC`) →
///   re-take the inner lock for the publish phase.
pub struct WriteConnection {
    db: Arc<Database>,
}

impl WriteConnection {
    pub(crate) fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    /// The owning `Database`.
    #[must_use]
    pub fn database(&self) -> &Arc<Database> {
        &self.db
    }

    // ---- delegated write methods (per-call write guard) ----

    pub fn create_collection(&mut self, name: impl Into<String>) -> Result<CollectionId> {
        self.db.inner.write().create_collection(name)
    }

    pub fn put_frame(&mut self, input: PutFrame<'_>) -> Result<FrameId> {
        self.db.inner.write().put_frame(input)
    }

    pub fn put_vector(&mut self, input: PutVector<'_>) -> Result<VectorId> {
        self.db.inner.write().put_vector(input)
    }

    pub fn delete_frame(&mut self, frame_id: FrameId) -> Result<()> {
        self.db.inner.write().delete_frame(frame_id)
    }

    pub fn delete_vector(&mut self, vector_id: VectorId) -> Result<()> {
        self.db.inner.write().delete_vector(vector_id)
    }

    pub fn flush(&mut self) -> Result<()> {
        self.db.inner.write().flush()
    }

    /// Commit pending mutations using Stage 5++'s avalanche
    /// pattern:
    ///
    /// 1. Take the `RwLock` write guard. Run phase A (segments,
    ///    footer, header, coord publish — all to page cache, no
    ///    fsync). Drop the guard.
    /// 2. Enter the pipeline's `GroupFsync` barrier. Multiple
    ///    concurrent committers pile in here while the current leader
    ///    is parked on F_FULLFSYNC; the next-promoted leader's fsync
    ///    covers all of them in one syscall.
    /// 3. Re-take the `RwLock` write guard. Run phase B (mmap remap,
    ///    snapshot publish). Drop.
    ///
    /// The window where phase A's pwrite-to-coord-region is
    /// page-cache-visible-but-not-yet-durable already exists in any
    /// fsync-batched system; it ends when the GroupFsync barrier
    /// returns. No lost-commit risk because durability is in the same
    /// fsync that covers the data.
    pub fn commit(&mut self) -> Result<CommitOutcome> {
        use crate::file::CommitProfile;

        let mut profile = CommitProfile::default();

        // Phase A: writes under the inner write guard. We also dup the
        // file fd here while we already hold the lock, so the fsync
        // step doesn't need to re-enter `inner` for any reason.
        let (prepared, cloned_file, path) = {
            let mut valise = self.db.inner.write();
            let prep = match valise.commit_phase_writes(&mut profile)? {
                crate::file::CommitWritePhaseResult::NoOp(outcome) => {
                    return Ok(outcome);
                }
                crate::file::CommitWritePhaseResult::Prepared(p) => p,
            };
            let (file, path) = valise.prepare_commit_fsync()?;
            (prep, file, path)
        };

        // **No `inner` lock and no `file` lock held here.** Concurrent
        // writers' phase A (which takes `inner.write()` + `file.lock()`
        // for ~30 µs) proceeds freely while this writer is parked on
        // the ~4 ms F_FULLFSYNC. Threads 2..N pile into the same
        // GroupFsync wave; the next leader's fsync covers all queued
        // tickets in one syscall. The cloned fd is dropped after the
        // closure returns.
        //
        // Stage 5++ optimization: only the file fsync runs here, NOT
        // `sync_parent_dir`. The directory entry is durable from
        // `create_with_options`'s setup; subsequent commits don't
        // mutate it. Skipping the per-commit dir fsync halves the
        // per-leader wall time on macOS APFS (each F_FULLFSYNC is a
        // separate hardware barrier).
        let _ = path;
        self.db.pipeline.commit_fsync.fsync(|| {
            crate::io::sync_file(&cloned_file, crate::io::Durability::FullSync)?;
            Ok(())
        })?;

        // Phase B: take the write guard again, do the publish work.
        let outcome = {
            let mut valise = self.db.inner.write();
            valise.commit_phase_publish(prepared, &mut profile)?
        };
        Ok(outcome)
    }

    // ---- delegated read methods (allowed within the writer's view) ----

    pub fn collections(&self) -> Vec<CollectionDesc> {
        self.db.inner.read().collections().to_vec()
    }

    pub fn frame_stubs(&self) -> Vec<FrameStub> {
        self.db.inner.read().frame_stubs().to_vec()
    }

    pub fn embedding_spaces(&self) -> Vec<EmbeddingSpaceDesc> {
        self.db.inner.read().embedding_spaces().to_vec()
    }

    pub fn frame_full(&self, frame_id: FrameId) -> Result<FrameDesc> {
        self.db.inner.read().frame_full(frame_id)
    }

    pub fn read_payload(&self, frame_id: FrameId) -> Result<Vec<u8>> {
        self.db.inner.read().read_payload(frame_id)
    }

    pub fn read_vector(
        &self,
        vector_id: VectorId,
        mode: crate::file::Reconstruct,
    ) -> Result<crate::file::ReadVectorResult> {
        self.db.inner.read().read_vector(vector_id, mode)
    }

    pub fn vector_search(&self, query: crate::file::VectorSearchQuery) -> Result<Vec<VectorHit>> {
        self.db.inner.read().vector_search(query)
    }

    pub fn snapshot(&self) -> Arc<Snapshot> {
        self.db.inner.read().snapshot()
    }
}
