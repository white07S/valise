// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

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
//!   the `Database`'s single-writer slot for the connection's lifetime,
//!   so `put_frame`/`commit` sequences from different threads cannot
//!   interleave. `Database::writer` blocks for the slot;
//!   `Database::try_writer` returns `None` instead of waiting.
//!
//! See `docs/CONCURRENCY.md`.

use std::path::PathBuf;
use std::sync::Arc;

use parking_lot::ArcMutexGuard;

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
/// # What the pin does and does not cover
///
/// Catalog reads (`collections`, `frame_stubs`, `vectors`, the profile and
/// schema tables) and payload reads (`frame_full`, `read_payload`,
/// `read_raw_text`) are served from the pinned `Snapshot` and are stable
/// for the connection's lifetime.
///
/// The **search methods are not pinned**: `query_text`, `query_hybrid`,
/// `vector_search`, `time_range_query`, `read_vector`, and the
/// `collection_member_*` accessors take the `Database`'s read lock and see
/// the latest committed state. `Snapshot` pins the mmap and the catalog,
/// but the lexical and vector indexes live in the `ValiseFile` and are not
/// snapshotted, so a query issued after a concurrent commit can return a
/// frame that this connection's own `frame_stubs()` does not list. Call
/// [`Self::refresh_snapshot`] to re-pin if you need the two to agree.
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

    // ---- catalog reads: served from the pinned snapshot ----
    //
    // These drive off `self.snapshot.catalog()`, which is the
    // `CatalogSnapshot` captured when this connection pinned. Reading them
    // through `db.inner` instead would return whatever a concurrent writer
    // has since committed, so a connection could observe its own frame
    // count change underneath it — the pin would be decorative.
    //
    // They take no lock, which is also why they are cheaper than the
    // query methods below.

    pub fn path(&self) -> PathBuf {
        self.db.inner.read().path().to_path_buf()
    }

    /// The live file header, **not** a pinned value. `Snapshot` captures
    /// `generation` and `toc_offset` but not the whole `Header`; use
    /// [`Snapshot::generation`] when you need the pinned identity.
    pub fn header(&self) -> Header {
        self.db.inner.read().header().clone()
    }

    pub fn collections(&self) -> Vec<CollectionDesc> {
        self.snapshot.catalog().collections.clone()
    }

    /// Heavy per-frame catalog as of the pin.
    ///
    /// Note this is the *decoded* `frames` vector, which the open
    /// fast-path deliberately leaves empty — see [`Self::frame_stubs`] for
    /// the always-populated `(frame_id, status)` view, and
    /// [`Self::frame_full`] to resolve one frame on demand.
    pub fn frames(&self) -> Vec<FrameDesc> {
        self.snapshot.catalog().frames.clone()
    }

    pub fn frame_stubs(&self) -> Vec<FrameStub> {
        self.snapshot.catalog().frame_stubs.clone()
    }

    /// Resolve a frame to its full `FrameDesc` via the pinned snapshot.
    /// **Does not lock anything** — drives directly off
    /// `Snapshot::frame_full`. Mixed-mode reads (concurrent with a
    /// committing writer) no longer wait on the writer's commit window.
    pub fn frame_full(&self, frame_id: FrameId) -> Result<FrameDesc> {
        self.snapshot.frame_full(frame_id)
    }

    pub fn vectors(&self) -> Vec<VectorDesc> {
        self.snapshot.catalog().vectors.clone()
    }

    pub fn embedding_spaces(&self) -> Vec<EmbeddingSpaceDesc> {
        self.snapshot.catalog().embedding_spaces.clone()
    }

    pub fn codecs(&self) -> Vec<CodecDesc> {
        self.snapshot.catalog().codecs.clone()
    }

    pub fn text_spaces(&self) -> Vec<TextSpaceDesc> {
        self.snapshot.catalog().text_spaces.clone()
    }

    pub fn analyzers(&self) -> Vec<AnalyzerDesc> {
        self.snapshot.catalog().analyzers.clone()
    }

    pub fn field_schemas(&self) -> Vec<FieldSchemaDesc> {
        self.snapshot.catalog().field_schemas.clone()
    }

    pub fn retrieval_profiles(&self) -> Vec<RetrievalProfileDesc> {
        self.snapshot.catalog().retrieval_profiles.clone()
    }

    pub fn fusion_profiles(&self) -> Vec<FusionProfileDesc> {
        self.snapshot.catalog().fusion_profiles.clone()
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

    /// Live bytes on disk per segment type: `(type, segment count, bytes)`,
    /// sorted by the type's wire discriminant.
    ///
    /// This is the "where did my file size go" accounting — payloads versus
    /// the term dictionary versus postings versus vector codes. Tombstoned
    /// segments are excluded, so after deletes the total is smaller than the
    /// file on disk until [`crate::db::Store::compact`] reclaims the gap.
    ///
    /// Cheap: one pass over the in-memory segment registry.
    pub fn storage_breakdown(&self) -> Vec<(crate::format::segment::SegmentType, u32, u64)> {
        self.db.inner.read().storage_breakdown_by_segment_type()
    }

    // ---- search and index reads: NOT pinned ----
    //
    // These take the read lock and observe the latest committed state.
    // The indexes they consult live in the `ValiseFile`, not in
    // `Snapshot`, so pinning them would mean snapshotting index state
    // too. See the type-level docs.

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

/// Writer handle. **At most one per `Database` at a time.**
///
/// - The connection holds the `Database`'s single-writer slot for its
///   whole lifetime, so a `put_frame`/`commit` sequence cannot interleave
///   with another writer's. [`Database::writer`] blocks to get here;
///   [`Database::try_writer`] returns `None` instead of waiting. Drop the
///   connection to release the slot.
///
///   This exclusion is load-bearing, not bookkeeping. Without it two
///   writers' puts interleave over the shared payload staging buffer and
///   the per-frame offset index that `commit` rewrites at flush, and two
///   frames end up pointing at the same bytes — a silently lost write.
///
/// - Writers do **not** block readers. Each write *method call* takes the
///   inner `RwLock` write guard, mutates, and releases it; readers
///   interleave freely between calls.
///
/// - **Puts are not isolated transactions.** There is no WAL — the v2
///   format removed it — so `put_frame` mutates the shared in-memory
///   catalog immediately and `commit()` is the durability point for
///   everything staged. A writer that puts and then drops without
///   committing leaves its rows staged for the next writer's commit. If
///   you need rollback, stage in your own buffer.
///
/// - `commit()` runs as a *two-phase avalanche*: phase A (writes, under
///   the inner `RwLock` write guard) → drop the lock → phase B's
///   GroupFsync barrier (no lock; multiple committers pile in here while
///   the leader is parked on the ~10 ms `F_FULLFSYNC`) → re-take the
///   inner lock for the publish phase. The barrier still coalesces
///   fsyncs across processes sharing the file.
pub struct WriteConnection {
    db: Arc<Database>,
    /// The `Database`'s single-writer slot, held for this connection's
    /// lifetime and released on drop. Never read — its value is the
    /// exclusion itself.
    _writer_slot: ArcMutexGuard<parking_lot::RawMutex, ()>,
}

impl WriteConnection {
    pub(crate) fn new(
        db: Arc<Database>,
        writer_slot: ArcMutexGuard<parking_lot::RawMutex, ()>,
    ) -> Self {
        Self {
            db,
            _writer_slot: writer_slot,
        }
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
