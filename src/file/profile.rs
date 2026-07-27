//! Commit-outcome and ingest/commit profiling value types for the
//! `ValiseFile` engine surface. Re-exported from `file.rs` so the public path
//! stays `crate::file::<Type>`.

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CommitOutcome {
    pub snapshot_generation: u64,
    pub changed: bool,
}

/// Per-phase per-call accumulators for the ingest hot path. Sampled
/// during the per-row loop, dumped + reset at user-chosen checkpoints
/// (typically every 10 000 rows from the bench harness) so the user
/// can see whether each phase scales linearly with corpus size.
///
/// Times are accumulated as nanoseconds in interior-mutability `Cell<u64>`
/// so the per-call instrumentation site is `let t = Instant::now();
/// f(); cell.set(cell.get() + t.elapsed().as_nanos())` — zero atomic
/// traffic, single-thread only.
///
/// Activation is **opt-in** via the `VALISE_INGEST_PROFILE` env var,
/// captured at `ValiseFile::create` / `open` time into a single `bool`.
/// When the flag is off, every instrumented site short-circuits.
#[derive(Debug, Default)]
pub struct IngestProfile {
    pub put_frame_calls: std::sync::atomic::AtomicU64,
    pub put_frame_collection_lookup_ns: std::sync::atomic::AtomicU64,
    pub put_frame_allocate_ids_ns: std::sync::atomic::AtomicU64,
    pub put_frame_segment_write_ns: std::sync::atomic::AtomicU64,
    pub put_frame_catalog_upsert_ns: std::sync::atomic::AtomicU64,
    pub put_frame_total_ns: std::sync::atomic::AtomicU64,

    pub idx_calls: std::sync::atomic::AtomicU64,
    pub idx_frame_lookup_ns: std::sync::atomic::AtomicU64,
    pub idx_text_space_lookup_ns: std::sync::atomic::AtomicU64,
    /// Sum of the three split phases below: `idx_read_payload_ns +
    /// idx_utf8_validate_ns + idx_analyze_total_ns`. Kept for back-compat
    /// with the original lumped counter.
    pub idx_analyzer_apply_ns: std::sync::atomic::AtomicU64,
    /// `self.read_payload(frame_id)` — segment lookup + seek + read +
    /// alloc of the payload bytes.
    pub idx_read_payload_ns: std::sync::atomic::AtomicU64,
    /// `std::str::from_utf8` validation of the payload.
    pub idx_utf8_validate_ns: std::sync::atomic::AtomicU64,
    /// Total time inside `Analyzer::analyze_with_breakdown` (sum of
    /// the three sub-fields below).
    pub idx_analyze_total_ns: std::sync::atomic::AtomicU64,
    /// Sub-phase 1: NFC / NFKC normalization.
    pub idx_analyze_normalize_ns: std::sync::atomic::AtomicU64,
    /// Sub-phase 2: UnicodeWords / Whitespace tokenization (UAX#29).
    pub idx_analyze_tokenize_ns: std::sync::atomic::AtomicU64,
    /// Sub-phase 3: per-token loop (fold + possessive-strip + stopword
    /// filter + Porter2 stem + length filter + collect to `Vec<u8>`).
    pub idx_analyze_token_loop_ns: std::sync::atomic::AtomicU64,
    /// Sum of raw tokens (post-tokenize, pre-filter) over all calls.
    /// Useful for normalizing the per-token-loop cost.
    pub idx_raw_tokens_total: std::sync::atomic::AtomicU64,
    /// Sum of output tokens (post-stopword/stem/length-filter) over all
    /// calls. Lower than raw_tokens when stopwords are enabled.
    pub idx_output_tokens_total: std::sync::atomic::AtomicU64,
    /// Total `StemCache::stem_into` calls aggregated across rayon
    /// workers (each call corresponds to one token reaching the stem
    /// step). Zero on the single-doc `index_frame_text` path.
    pub idx_stem_cache_lookups: std::sync::atomic::AtomicU64,
    /// Subset of `idx_stem_cache_lookups` that found a hit.
    pub idx_stem_cache_hits: std::sync::atomic::AtomicU64,
    /// Subset that bypassed the cache (token didn't fit in 31 bytes,
    /// or the stem result didn't). Zero for English BEIR.
    pub idx_stem_cache_bypassed: std::sync::atomic::AtomicU64,
    pub idx_pending_build_ns: std::sync::atomic::AtomicU64,
    pub idx_pending_insert_ns: std::sync::atomic::AtomicU64,
    pub idx_total_ns: std::sync::atomic::AtomicU64,

    pub vec_calls: std::sync::atomic::AtomicU64,
    pub vec_frame_lookup_ns: std::sync::atomic::AtomicU64,
    pub vec_space_lookup_ns: std::sync::atomic::AtomicU64,
    pub vec_codec_get_ns: std::sync::atomic::AtomicU64,
    pub vec_encode_ns: std::sync::atomic::AtomicU64,
    pub vec_pending_append_ns: std::sync::atomic::AtomicU64,
    /// In-memory catalog upsert of the `VectorDesc` + dirty-tracking
    /// insert. (Formerly `vec_wal_buffer_ns`; v2 has no WAL.)
    pub vec_catalog_upsert_ns: std::sync::atomic::AtomicU64,
    pub vec_total_ns: std::sync::atomic::AtomicU64,
}

impl IngestProfile {
    /// Reset every counter to zero. Called after each checkpoint so the
    /// next window's averages reflect only that window's calls. The
    /// counters are `AtomicU64` (Stage 1: required for `ValiseFile: Sync`);
    /// every site uses `Relaxed` ordering since the data is profiling
    /// only, not a synchronization signal.
    pub fn reset(&self) {
        use std::sync::atomic::Ordering::Relaxed;
        macro_rules! z { ($($f:ident),* $(,)?) => { $( self.$f.store(0, Relaxed); )* } }
        z!(
            put_frame_calls,
            put_frame_collection_lookup_ns,
            put_frame_allocate_ids_ns,
            put_frame_segment_write_ns,
            put_frame_catalog_upsert_ns,
            put_frame_total_ns,
            idx_calls,
            idx_frame_lookup_ns,
            idx_text_space_lookup_ns,
            idx_analyzer_apply_ns,
            idx_read_payload_ns,
            idx_utf8_validate_ns,
            idx_analyze_total_ns,
            idx_analyze_normalize_ns,
            idx_analyze_tokenize_ns,
            idx_analyze_token_loop_ns,
            idx_raw_tokens_total,
            idx_output_tokens_total,
            idx_stem_cache_lookups,
            idx_stem_cache_hits,
            idx_stem_cache_bypassed,
            idx_pending_build_ns,
            idx_pending_insert_ns,
            idx_total_ns,
            vec_calls,
            vec_frame_lookup_ns,
            vec_space_lookup_ns,
            vec_codec_get_ns,
            vec_encode_ns,
            vec_pending_append_ns,
            vec_catalog_upsert_ns,
            vec_total_ns,
        );
    }
}

/// Per-step breakdown of the in-memory encoding work inside
/// `build_flush_output`. The total of these fields equals
/// `CommitProfile::text_indexes_build` (modulo a few nanoseconds of
/// bookkeeping). Field names map 1:1 to the numbered comments inside
/// `src/file/text_indexing.rs::build_flush_output`.
#[derive(Clone, Copy, Debug, Default)]
pub struct TextIndexBuildProfile {
    /// Step 1: BTreeMap<Vec<u8>, ()> of newly-introduced term bytes.
    pub collect_new_terms: std::time::Duration,
    /// Step 2: assign monotonic term ids.
    pub assign_term_ids: std::time::Duration,
    /// Step 3: per-term posting list deltas (BTreeMap<TermId, Vec<Posting>>).
    pub build_postings: std::time::Duration,
    /// Step 4: per-term-dict-record df/cf aggregation.
    pub build_term_dict: std::time::Duration,
    /// Step 5: update running df/cf and per-term posting state.
    pub update_state: std::time::Duration,
    /// Step 6: bincode encode the three delta segments.
    pub encode_segments: std::time::Duration,
}

/// Per-phase wall-clock breakdown emitted by [`ValiseFile::commit_with_profile`].
///
/// Captured live (not via env vars or logging) so the bench harness — and
/// any other caller that wants to understand commit-time costs — can print
/// or aggregate them as structured data. Phases are recorded in the order
/// the commit() implementation visits them; sums of the listed phases plus
/// `header_write` + `mmap_remap` + `by_id_cache_rebuild` should approximate
/// `total` (small bookkeeping deltas excluded).
#[derive(Clone, Copy, Debug, Default)]
pub struct CommitProfile {
    /// `flush()` of the Buffered-mode page cache before the snapshot.
    pub flush: std::time::Duration,
    /// Vector batch encode + segment append (QAM codes for all pending vectors).
    pub vector_batches: std::time::Duration,
    /// Total time inside `flush_pending_text_indexes` (sum of build + write).
    pub text_indexes: std::time::Duration,
    /// Sub-phase: term dict / postings / docstats *encoding* (in-memory build).
    pub text_indexes_build: std::time::Duration,
    /// Per-step breakdown inside `build_flush_output`. Only populated when
    /// the text-indexing path runs.
    pub text_indexes_build_breakdown: TextIndexBuildProfile,
    /// Sub-phase: appending those three segments to disk (BLAKE3 + write).
    pub text_indexes_write: std::time::Duration,
    /// Sum of all `flush_catalog_table` calls (collection / frame / text_space /
    /// analyzer / field_schema / retrieval_profile / embedding_space / codec /
    /// vector / fusion_profile).
    pub catalog_tables: std::time::Duration,
    /// Per-collection roaring-bitmap-style filter rebuild.
    pub collection_filters: std::time::Duration,
    /// Global time-index segment rebuild.
    pub time_index: std::time::Duration,
    /// Segment-registry spine append (the only segment not self-listed).
    pub segment_registry: std::time::Duration,
    /// `fsync` after every appended segment is on disk.
    pub fsync_segments: std::time::Duration,
    /// TOC body encode + footer write + footer fsync.
    pub footer_write: std::time::Duration,
    /// Header rewrite (`footer_offset` + checksum + generation) + its fsync.
    pub header_write: std::time::Duration,
    /// `mmap` remap over the now-larger file.
    pub mmap_remap: std::time::Duration,
    /// Refresh of `vector_by_id_cache` from the new committed catalog.
    pub by_id_cache_rebuild: std::time::Duration,
    /// Wall-clock from commit() entry to commit() exit.
    pub total: std::time::Duration,
}
