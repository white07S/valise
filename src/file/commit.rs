// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `ValiseFile` commit pipeline: flush / commit / coordination / segment writers.

use super::*;

impl ValiseFile {
    /// Force any segment bytes that have been written under
    /// [`Durability::Buffered`] to durable storage. No-op for per-call
    /// durability modes (their per-call fsyncs already covered it).
    ///
    /// This is *not* a snapshot. The active TOC footer is unchanged, so
    /// the flushed bytes are orphan appends past the current
    /// `footer_offset` until a `commit()` publishes a new footer that
    /// references them; a reopen before that commit ignores them. To
    /// produce a new snapshot (commit a TOC update), call `commit`.
    ///
    /// `commit()` issues its own trailing fsync (the avalanche barrier),
    /// so callers that always commit between batches do not need to
    /// invoke this directly.
    pub fn flush(&mut self) -> Result<()> {
        self.ensure_write()?;
        if self.durability != Durability::Buffered {
            return Ok(());
        }
        // One F_FULLFSYNC drains the file's page cache: every segment
        // byte appended since the last barrier becomes durable atomically
        // from the next reader's perspective. Recovery is footer-driven
        // (read header → read TOC), so any orphan tail past the active
        // `footer_offset` that didn't make it to disk is simply ignored.
        sync_file(&self.file.lock(), Durability::FullSync)?;
        sync_parent_dir(&self.path, Durability::FullSync)?;
        Ok(())
    }

    pub fn commit(&mut self) -> Result<CommitOutcome> {
        self.commit_with_profile().map(|(o, _)| o)
    }

    /// Commit + return a per-phase wall-clock breakdown. Same effects as
    /// `commit`; the second tuple element captures where time was spent.
    /// Useful when you need to understand commit-time costs without
    /// instrumenting the library or shelling out to a profiler.
    pub fn commit_with_profile(&mut self) -> Result<(CommitOutcome, CommitProfile)> {
        self.ensure_write()?;
        let mut profile = CommitProfile::default();
        let total_start = std::time::Instant::now();

        match self.commit_phase_writes(&mut profile)? {
            CommitWritePhaseResult::NoOp(outcome) => {
                profile.total = total_start.elapsed();
                Ok((outcome, profile))
            }
            CommitWritePhaseResult::Prepared(prepared) => {
                self.run_commit_fsync()?;
                let outcome = self.commit_phase_publish(prepared, &mut profile)?;
                profile.total = total_start.elapsed();
                Ok((outcome, profile))
            }
        }
    }

    /// Stage 5++ avalanche helper. Runs the dirty-check + write phase,
    /// returns prepared state for `commit_phase_publish` to consume
    /// after the fsync barrier. The orchestrator (either
    /// `commit_with_profile` for the legacy single-thread path or
    /// `WriteConnection::commit` for the avalanche path) is
    /// responsible for sequencing this call → fsync → publish.
    ///
    /// # Crash safety contract
    ///
    /// This function performs THREE logical writes in user-space order:
    ///
    ///   1. New segments appended at EOF (payloads, vector data,
    ///      catalog deltas, text indexes).
    ///   2. New TOC footer appended at EOF — self-validating via the
    ///      embedded body and footer checksums (see
    ///      [`crate::format::toc::TocFooterCodec`]).
    ///   3. Header rewrite of the 120-byte logical prefix —
    ///      `footer_offset` and `snapshot_generation` updated together.
    ///      The aligned 8-byte `footer_offset` store within that write
    ///      is the atomic commit switch; the surrounding prefix can
    ///      still tear, which is why `resolve_footer_state` cross-checks
    ///      the header generation against the footer's and falls back to
    ///      a scan when they disagree.
    ///
    /// All three writes go to the OS page cache under
    /// `Durability::Buffered` (no per-step fsync). Durability comes
    /// from a SINGLE `F_FULLFSYNC` issued by the orchestrator
    /// (`run_commit_fsync`) AFTER this function returns. That single
    /// barrier makes the whole commit atomic at the hardware level —
    /// either every byte written here is durable (new commit visible)
    /// or none are (old commit visible). There is no WAL: recovery is
    /// driven entirely from the header → footer chain (the v2 format
    /// removed the embedded WAL; see `MIGRATION.md`).
    ///
    /// Crash recovery scenarios this protocol handles correctly,
    /// validated by
    /// `tests/qam_vector.rs::crash_safety_open_after_partial_segment_truncation`:
    ///
    ///   - **Crash before `run_commit_fsync`**: any subset of the three
    ///     writes may be in flight in the page cache. Filesystem
    ///     writeback ordering (APFS, ext4 with `data=ordered`) means
    ///     write N reaches disk only after writes 1..N-1, so the
    ///     observable post-crash state is always a PREFIX of the three
    ///     writes. With nothing or only segments/TOC durable, the OLD
    ///     header still points to the OLD footer offset and recovery
    ///     uses the prior snapshot. Orphan bytes past the old
    ///     `footer_offset` are invisible to the segment registry.
    ///
    ///   - **Crash after `run_commit_fsync`**: every byte from this
    ///     commit is durable; recovery uses the new header → new TOC →
    ///     new snapshot.
    ///
    ///   - **Torn TOC (partial write of step 2)**: the embedded TOC
    ///     checksums fail at `read_toc_footer`; recovery surfaces an
    ///     explicit error rather than silently accepting a half-state.
    pub(crate) fn commit_phase_writes(
        &mut self,
        profile: &mut CommitProfile,
    ) -> Result<CommitWritePhaseResult> {
        self.ensure_write()?;

        if !self.dirty {
            return Ok(CommitWritePhaseResult::NoOp(CommitOutcome {
                snapshot_generation: self.header.snapshot_generation,
                changed: false,
            }));
        }

        // Stage 3b: take the writer-slot exclusive byte lock. Single
        // writer at a time across processes for a given file. We hold
        // the lock across all disk writes + the coord publish, then
        // release in phase B (`commit_phase_publish`).
        let writer_lock_acquired = self.coord_acquire_writer_lock()?;

        // Stage 5++ avalanche: SKIP the historical mid-commit
        // `self.flush()` (which under Buffered durability did a full
        // F_FULLFSYNC over prior put_frame traffic, ~3.7 ms on this
        // APFS device). The trailing `run_commit_fsync` covers the
        // same byte range — that early fsync was a redundant barrier
        // for "surface page-cache errors before we write the TOC",
        // which collapses to "surface them at the trailing fsync"
        // for our purposes. Cuts the per-commit cost from 2× APFS
        // hardware barriers (~8 ms) to 1× (~4 ms).
        profile.flush = Default::default();

        // Flush the per-commit payload batch before any catalog write.
        // The flushed segment must be in the registry by the time
        // `write_catalog_segments_profiled` snapshots it, otherwise
        // the FrameCatalog will reference a segment that hasn't been
        // appended yet.
        self.flush_pending_payloads()?;

        let t = std::time::Instant::now();
        self.flush_pending_vector_batches()?;
        profile.vector_batches = t.elapsed();

        // Stage 5++ fsync coalescing: do all writes (segments, footer,
        // coord publish, header) under
        // `Durability::Buffered` so the per-step `sync_file` calls
        // become no-ops, then issue ONE `FullSync` at the end of the
        // write phase. On macOS APFS each `F_FULLFSYNC` is a hardware
        // barrier (~5–25 ms even when there are no fresh dirty pages);
        // collapsing four into one cuts a large constant out of every
        // commit. The kernel page cache preserves write order at the
        // file level; the `coord_published_*` atomics are pwritten
        // BEFORE the fsync so the post-fsync read of those atomics is
        // guaranteed to see durable bytes.
        let buffered = Durability::Buffered;
        let body =
            self.write_catalog_segments_profiled(self.header.snapshot_generation + 1, profile)?;
        let t = std::time::Instant::now();
        sync_file(&self.file.lock(), buffered)?; // no-op
        profile.fsync_segments = t.elapsed();

        let t = std::time::Instant::now();
        // Single canonical wire shape — the contract round-trips
        // unchanged on every commit; the digest in the header anchors
        // it as immutable.
        let footer = TocFooterCodec::encode_body(body)?;
        let footer_offset = {
            let file_guard = self.file.lock();
            let meta = file_guard.metadata()?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::MetadataExt;
                if meta.nlink() == 0 {
                    drop(file_guard); // MUST drop before coord_release_writer_lock (it re-locks self.file)
                    self.coord_release_writer_lock();
                    return Err(Error::Busy(
                        "commit aborted: the file was replaced by a concurrent compaction (stale inode); reopen the Store".into(),
                    ));
                }
            }
            meta.len()
        };
        write_toc_footer_at_end(&mut self.file.lock(), &footer)?;
        sync_file(&self.file.lock(), buffered)?; // no-op
        profile.footer_write = t.elapsed();

        let t = std::time::Instant::now();
        self.header.footer_offset = footer_offset;
        self.header.snapshot_generation = footer.body.snapshot_generation;
        self.coord_publish(footer_offset, footer.body.snapshot_generation)?;
        HeaderCodec::write(&mut *self.file.lock(), &self.header)?;
        sync_file(&self.file.lock(), buffered)?; // no-op
        profile.header_write = t.elapsed();

        // Stage 5++ avalanche: capture each writer's per-commit state
        // here, BEFORE dropping the inner write lock. Phase B uses the
        // captured snapshot rather than reading `self.header` directly,
        // so concurrent writers' header rewrites don't corrupt each
        // other's published snapshots.
        let vectors_changed = !self.dirty_vector_ids.is_empty()
            || !self.dirty_codec_ids.is_empty()
            || !self.dirty_embedding_space_ids.is_empty();
        Ok(CommitWritePhaseResult::Prepared(CommitPhaseAState {
            footer,
            captured_header: self.header.clone(),
            writer_lock_acquired,
            vectors_changed,
        }))
    }

    /// Stage 5++ phase B. Runs after `run_commit_fsync` returns. Updates
    /// in-memory state, remaps the mmap, publishes the new snapshot.
    /// Uses the snapshot captured in phase A instead of reading
    /// `self.header` so concurrent writers don't corrupt each other's
    /// published states.
    pub(crate) fn commit_phase_publish(
        &mut self,
        prepared: CommitPhaseAState,
        profile: &mut CommitProfile,
    ) -> Result<CommitOutcome> {
        let CommitPhaseAState {
            footer,
            captured_header,
            writer_lock_acquired,
            vectors_changed,
        } = prepared;

        self.footer = footer;
        self.dirty_segment_ids.clear();
        self.dirty_collection_ids.clear();
        self.dirty_frame_ids.clear();
        self.dirty_embedding_space_ids.clear();
        self.dirty_codec_ids.clear();
        self.dirty_vector_ids.clear();
        self.dirty_text_space_ids.clear();
        self.dirty_analyzer_ids.clear();
        self.dirty_field_schema_ids.clear();
        self.dirty_retrieval_profile_ids.clear();
        self.dirty_fusion_profile_ids.clear();
        self.dirty = false;

        let t = std::time::Instant::now();
        self.file_mmap = remap_file(&self.file.lock())?.map(Arc::new);
        profile.mmap_remap = t.elapsed();

        let t = std::time::Instant::now();
        if vectors_changed {
            self.vector_by_id_cache = rebuild_vector_by_id_cache(&self.catalog);
            let registry = self.segment_registry.read();
            let mut vb = self.vector_base.write();
            if let Some(ref mmap) = self.file_mmap {
                let (
                    ptrs,
                    stride,
                    candidates_by_space,
                    qam_norms_by_space,
                    sketches_by_space,
                    upq_i8_by_space,
                    upq_dequant_by_space,
                ) = build_vector_base_ptrs(
                    &self.catalog,
                    &registry.by_id,
                    mmap,
                    &self.codec_cache,
                    None,
                )?;
                vb.ptrs = ptrs;
                vb.stride = stride;
                vb.candidates_by_space = candidates_by_space;
                vb.qam_norms_by_space = qam_norms_by_space;
                vb.sketches_by_space = sketches_by_space;
                vb.upq_i8_by_space = upq_i8_by_space;
                vb.upq_dequant_by_space = upq_dequant_by_space;
                vb.loaded = true;
            } else {
                vb.invalidate();
            }
        } else {
            // The remap above happened regardless of whether any vector
            // changed, so every address in `vector_base` is now dangling
            // even though the vector *data* is untouched. Rebuilding here
            // would make text-only commits pay for a vector index they
            // did not modify, so instead drop the cache and let the next
            // vector search lazy-load it against the new mapping via
            // `ensure_vector_base_loaded`.
            //
            // Skipping this is a use-after-free. It is reachable from any
            // file that holds vectors and takes a commit touching only
            // text or payloads — two collections where one has no vector
            // space is the ordinary way to get there.
            self.vector_base.write().invalidate();
        }
        profile.by_id_cache_rebuild = t.elapsed();

        // Build the snapshot off the *captured* header so concurrent
        // writers' phase-A header mutations don't leak into our
        // published snapshot.
        let new_snapshot = build_snapshot(
            &captured_header,
            self.file_mmap.clone(),
            &self.catalog,
            &self.frame_locators,
            &self.segment_registry.read(),
            &self.vector_by_id_cache,
            self.verified_payload_segments.clone(),
        );
        self.published_snapshot.store(Arc::new(new_snapshot));

        if writer_lock_acquired {
            self.coord_release_writer_lock();
        }

        Ok(CommitOutcome {
            snapshot_generation: captured_header.snapshot_generation,
            changed: true,
        })
    }

    /// Stage 5++ commit-fsync hook (legacy / single-thread `ValiseFile`
    /// path). Routes through the optional `GroupFsync` barrier when
    /// one's installed; falls back to a local `FullSync`. The
    /// `WriteConnection` avalanche path bypasses this entirely — see
    /// `prepare_commit_fsync` below.
    pub(crate) fn run_commit_fsync(&self) -> Result<()> {
        let do_local = || -> Result<()> {
            let cloned_file = self.file.lock().try_clone()?;
            sync_file(&cloned_file, Durability::FullSync)?;
            sync_parent_dir(&self.path, Durability::FullSync)?;
            Ok(())
        };
        if let Some(barrier) = self.commit_fsync_barrier.read().as_ref() {
            barrier.fsync(do_local)
        } else {
            do_local()
        }
    }

    /// Snapshot the data needed to run the commit fsync **without
    /// holding any `ValiseFile`-level lock**. Used by the
    /// `WriteConnection` avalanche path: dup the file fd + clone the
    /// path, return both, then the caller drops every guard and
    /// invokes the GroupFsync barrier with no `RwLock`/`Mutex` held
    /// across the ~4 ms hardware barrier. While this writer is
    /// parked on F_FULLFSYNC, concurrent writers freely take
    /// `Database::inner.write()` for their own phase-A work and
    /// pile into the same coalescing wave.
    pub(crate) fn prepare_commit_fsync(&self) -> Result<(std::fs::File, std::path::PathBuf)> {
        let cloned_file = self.file.lock().try_clone()?;
        Ok((cloned_file, self.path.clone()))
    }

    /// Stage 5++: route this `ValiseFile`'s commit-fsync through the
    /// supplied [`crate::concurrency::writer_pipeline::GroupFsync`].
    /// `Database` calls this once after construction so subsequent
    /// commits coalesce.
    pub(crate) fn install_commit_fsync_barrier(
        &self,
        barrier: Arc<crate::concurrency::writer_pipeline::GroupFsync>,
    ) {
        *self.commit_fsync_barrier.write() = Some(barrier);
    }

    // ---- Stage 3b coordination helpers -------------------------------------

    /// Acquire the writer-slot exclusive byte lock. No-op + returns
    /// `false` when the file's coord region is inactive (legacy v0.1).
    /// Returns `true` when the lock is held; the caller must call
    /// `coord_release_writer_lock` after publish.
    pub(crate) fn coord_acquire_writer_lock(&self) -> Result<bool> {
        if !self.coord_region_active() {
            return Ok(false);
        }
        let file_guard = self.file.lock();
        let fd = std::os::fd::AsFd::as_fd(&*file_guard);
        // Bounded blocking with backoff. 1024 attempts ≈ several seconds
        // at the spin cap (2 ms); long enough that genuine contention
        // resolves, short enough to surface deadlocked writers as
        // `Error::Busy` instead of hanging.
        crate::concurrency::locks::acquire_exclusive_blocking(
            fd,
            crate::concurrency::coordination::WRITER_LOCK_BYTE,
            1024,
        )?;
        Ok(true)
    }

    /// Release the writer-slot exclusive byte lock acquired via
    /// `coord_acquire_writer_lock`. Best-effort — release errors are
    /// ignored because they cannot fail in a way the caller can recover
    /// from at this point in the commit flow (the bytes are already
    /// durable).
    pub(crate) fn coord_release_writer_lock(&self) {
        let file_guard = self.file.lock();
        let fd = std::os::fd::AsFd::as_fd(&*file_guard);
        let _ = crate::concurrency::locks::release_byte_lock(
            fd,
            crate::concurrency::coordination::WRITER_LOCK_BYTE,
        );
    }

    /// Publish the new TOC offset + snapshot generation into the coord
    /// region. Uses `pwrite` instead of mmap atomic stores: the file is
    /// mapped `PROT_READ` for the data path, so writing through the
    /// mmap would `SIGBUS`. `pwrite` of 8-byte aligned values is
    /// atomic at the page-cache level on every supported target — the
    /// reader's `Acquire` mmap load sees either the old or the new
    /// value, never a torn intermediate. The toc_offset is published
    /// before the generation so a reader that observes the new
    /// generation is guaranteed to see the matching offset.
    /// No-op when the coord region is inactive (legacy file).
    pub(crate) fn coord_publish(&self, toc_offset: u64, snapshot_generation: u64) -> Result<()> {
        if !self.coord_region_active() {
            return Ok(());
        }
        use std::os::unix::fs::FileExt;
        let toc_off_at = (crate::format::COORD_REGION_OFFSET
            + crate::concurrency::coordination::header_offset::PUBLISHED_TOC_OFFSET)
            as u64;
        let gen_at = (crate::format::COORD_REGION_OFFSET
            + crate::concurrency::coordination::header_offset::PUBLISHED_SNAPSHOT_GENERATION)
            as u64;
        let file = self.file.lock();
        // Order: TOC offset first, then generation. A reader doing
        // `Acquire` load on generation and seeing the new value is
        // guaranteed (via fsync below) to see the new TOC offset too.
        file.write_at(&toc_offset.to_le_bytes(), toc_off_at)?;
        file.write_at(&snapshot_generation.to_le_bytes(), gen_at)?;
        Ok(())
    }

    /// `true` when the file's coord region is recognizable (magic +
    /// version match this build). Drives the legacy-vs-Stage-3b dispatch
    /// at writer entry.
    pub(crate) fn coord_region_active(&self) -> bool {
        let Some(mmap) = self.file_mmap.as_ref() else {
            return false;
        };
        let Some(region) = crate::concurrency::coordination::region_slice(mmap) else {
            return false;
        };
        crate::concurrency::coordination::is_active(region)
    }

    /// Read the most recently published snapshot generation directly
    /// from the coord region. Used by tests + cross-process readers as
    /// the canonical visible commit point. Returns `None` for legacy
    /// files; `Some(0)` for new files pre-first-commit.
    pub fn coord_published_generation(&self) -> Option<u64> {
        let mmap = self.file_mmap.as_ref()?;
        let region = crate::concurrency::coordination::region_slice(mmap)?;
        if !crate::concurrency::coordination::is_active(region) {
            return None;
        }
        use std::sync::atomic::Ordering;
        // SAFETY: see coord_publish.
        let generation = unsafe {
            crate::concurrency::coordination::published_snapshot_generation(region)
                .load(Ordering::Acquire)
        };
        Some(generation)
    }

    pub(crate) fn ensure_write(&self) -> Result<()> {
        if self.mode != OpenMode::ReadWrite {
            return Err(Error::Unsupported("file opened read-only".into()));
        }
        // Write paths register segments → must observe the on-disk
        // segment registry first (Exploit A2 ghost registry: loaded
        // lazily on the first write rather than eagerly at open).
        self.ensure_segment_registry_loaded()
    }

    /// Phase 1 contract gate. Returns `Error::Unsupported` when the
    /// file was created with `TextMode::Disabled`. Called from every
    /// public text register/query entry point so a misconfigured
    /// caller hits the contract violation BEFORE any state mutates.
    pub(crate) fn ensure_text_enabled(&self, op: &'static str) -> Result<()> {
        if !self.create_contract.text_enabled {
            return Err(Error::Unsupported(format!(
                "{op}: file was created with TextMode::Disabled"
            )));
        }
        Ok(())
    }

    /// Ensure the segment registry has been loaded (Exploit A2 ghost
    /// registry). The open path defers `read_segment_catalog`; any code
    /// path that needs the by-id index (vector reads, vector_search,
    /// post-write segment registration) calls this first to lazy-load.
    /// Cheap no-op when already loaded. Stage 1:
    /// the lazy-load itself happens under the registry's write lock so
    /// the helper takes `&self`.
    pub(crate) fn ensure_segment_registry_loaded(&self) -> Result<()> {
        // Fast path: already loaded.
        if self.segment_registry.read().loaded {
            return Ok(());
        }
        // Slow path: take the write lock and check again under the
        // exclusive guard before doing the file read.
        let mut guard = self.segment_registry.write();
        if guard.loaded {
            return Ok(());
        }
        let cat = read_segment_catalog(&mut self.file.lock(), &self.footer.body)?;
        let by_id = build_segment_map(&cat);
        guard.catalog = cat;
        guard.by_id = by_id;
        guard.loaded = true;
        Ok(())
    }

    /// Lazy build of the naked-pointer vector base cache. First call
    /// loads the segment registry (if not already), then walks
    /// `catalog.vectors` and stores the absolute mmap addresses of each
    /// active vector's base record. Idempotent / cheap no-op once
    /// loaded; invalidated by `commit()` (which rebuilds in lockstep
    /// with the mmap remap).
    pub(crate) fn ensure_vector_base_loaded(&self) -> Result<()> {
        if self.vector_base.read().loaded {
            return Ok(());
        }
        self.ensure_segment_registry_loaded()?;
        let mut guard = self.vector_base.write();
        if guard.loaded {
            return Ok(());
        }
        if let Some(ref mmap) = self.file_mmap {
            let registry = self.segment_registry.read();
            let (
                ptrs,
                stride,
                candidates_by_space,
                qam_norms_by_space,
                sketches_by_space,
                upq_i8_by_space,
                upq_dequant_by_space,
            ) = build_vector_base_ptrs(
                &self.catalog,
                &registry.by_id,
                mmap,
                &self.codec_cache,
                Some(&self.verified_payload_segments),
            )?;
            guard.ptrs = ptrs;
            guard.stride = stride;
            guard.candidates_by_space = candidates_by_space;
            guard.qam_norms_by_space = qam_norms_by_space;
            guard.sketches_by_space = sketches_by_space;
            guard.upq_i8_by_space = upq_i8_by_space;
            guard.upq_dequant_by_space = upq_dequant_by_space;
        }
        guard.loaded = true;
        Ok(())
    }
}
