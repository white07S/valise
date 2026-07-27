//! Text-indexing lifecycle helpers (Phase 2 Group D).
//!
//! Maintains the per-text-space in-memory state (`TextSpaceState`), buffers
//! `index_frame_text` calls into a pending map, and at commit-time flushes
//! delta `TermDictionarySegment` / `PostingsSegment` / `DocStatsSegment`
//! segments. At open-time, rebuilds `TextSpaceState` by walking each
//! per-text-space chain oldest-to-newest.
//!
//! Per spec §8.2, no WAL op is needed: the canonical primitives are derived
//! from durable frame payloads, so a crash before commit just loses the
//! in-memory pending buffer; the user re-calls `index_frame_text`.
//!
//! v1 simplifications (documented in the public `index_frame_text` API):
//! - Source text comes from `frame.payload` (UTF-8 expected).
//!   `search_text_ref` resolution is v2.
//! - Single-field schema: `field_lengths = vec![total_token_count]`.
//!   Multi-field indexing per `FieldSchemaDesc.fields[]` is v2.
//! - Frame deletion does not rewrite postings (§12.0.3). Tombstoned frames
//!   remain in postings; queries filter at result time.

use std::{
    collections::{BTreeSet, HashMap},
    fs::File,
    sync::Arc,
};

use parking_lot::Mutex;
use rustc_hash::FxHashMap;

use crate::{
    error::{Error, Result},
    format::{
        CollectionId, FrameId, TermId, TextSpaceId,
        doc_stats::{DocStatsSegment, decode_doc_stats_segment, encode_doc_stats_segment},
        postings::{
            PostingsSegment, PostingsSegmentDirectory, decode_postings_segment_directory,
            decode_term_blob_into, encode_postings_segment,
        },
        registry::IdAllocator,
        segment::{SegmentRef, SegmentType},
        term_dictionary::{TermDictionarySegment, encode_term_dictionary_segment},
        text::{DocStatsEntry, PositionsRef, Posting, PostingList, TermDictionaryEntry},
        toc::{TextIndexRoot, TocFooterBody},
    },
};

use super::segment_io::read_segment_payload_typed;

/// Struct-of-Arrays representation of one term's posting list, kept sorted
/// ascending by `frame_id`. The hot retrieval loop only touches `frame_ids`
/// and `term_freqs`; cold metadata (`collection_ids`, `field_masks`,
/// `positions_refs`) lives in parallel arrays so cache lines on the hot path
/// pack ~8–16 entries instead of 1–2.
///
/// `positions_refs` is `None` for terms whose `AnalyzerDesc.store_positions`
/// is false — the universal v1 case. When `Some`, the parallel vector
/// matches the length of `frame_ids` exactly.
#[derive(Clone, Debug, Default)]
pub(crate) struct PostingListSoA {
    pub(crate) frame_ids: Vec<u64>,
    pub(crate) term_freqs: Vec<u32>,
    pub(crate) collection_ids: Vec<u32>,
    pub(crate) field_masks: Vec<u32>,
    pub(crate) positions_refs: Option<Vec<PositionsRef>>,
}

/// Pointer at one term's posting blob within `state.postings_segments`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PostingsLocator {
    pub(crate) segment_idx: u32,
    pub(crate) entry_idx: u32,
}

/// Reference to a term's bytes inside `TextSpaceState::term_dict_blobs`.
/// `(blob_idx, start, length, term_id)` — no owned `Vec<u8>`.
/// Replaces the per-term `Vec<u8>` allocation in `sorted_terms`.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TermSlice {
    pub(crate) blob_idx: u32,
    pub(crate) start: u32,
    pub(crate) length: u32,
    pub(crate) term_id: TermId,
}

/// Sentinel for `single_segment_dir_index` slots that don't carry a
/// posting list (e.g., a term in the dictionary with no postings yet).
pub(crate) const NO_POSTING_ENTRY: u32 = u32::MAX;

/// Lazy view of one term's posting list. **`Sparse`** is varint-decoded
/// into the canonical SoA arrays. **`Dense`** is the Roaring fast-path:
/// the underlying blob is a packed `[u64]` bitset (1 bit per frame_id),
/// borrowed straight out of the segment's mmap-retained buffer. The
/// dense scorer iterates set bits via `trailing_zeros` + `x & (x-1)` and
/// treats `tf=1` for every frame.
pub(crate) enum LazyPostingList<'a> {
    Sparse(crate::format::postings::PostingListSoaDecoded),
    Dense {
        term_id: TermId,
        df: u32,
        bitset: &'a [u8],
    },
}

impl PostingListSoA {
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.frame_ids.len()
    }

    pub(crate) fn reserve(&mut self, additional: usize) {
        self.frame_ids.reserve(additional);
        self.term_freqs.reserve(additional);
        self.collection_ids.reserve(additional);
        self.field_masks.reserve(additional);
        if let Some(refs) = &mut self.positions_refs {
            refs.reserve(additional);
        }
    }

    /// Bulk-append a frame-ascending posting list when the caller guarantees
    /// Insert or replace the entry for `posting.frame_id`, preserving
    /// ascending frame_id order (per spec §12.0.4 newest-wins semantics for
    /// re-index of the same frame).
    pub(crate) fn upsert(&mut self, posting: Posting) {
        match self.frame_ids.binary_search(&posting.frame_id.0) {
            Ok(idx) => {
                self.term_freqs[idx] = posting.term_freq;
                self.collection_ids[idx] = posting.collection_id.0;
                self.field_masks[idx] = posting.field_mask;
                self.upsert_positions_ref(idx, posting.positions_ref, true);
            }
            Err(idx) => {
                self.frame_ids.insert(idx, posting.frame_id.0);
                self.term_freqs.insert(idx, posting.term_freq);
                self.collection_ids.insert(idx, posting.collection_id.0);
                self.field_masks.insert(idx, posting.field_mask);
                self.upsert_positions_ref(idx, posting.positions_ref, false);
            }
        }
    }

    fn upsert_positions_ref(
        &mut self,
        idx: usize,
        new_ref: Option<PositionsRef>,
        replace_existing: bool,
    ) {
        match (&mut self.positions_refs, new_ref) {
            (None, None) => {}
            (None, Some(r)) => {
                // First positions_ref encountered for this term — materialize
                // the parallel vec to match `frame_ids` (which already
                // includes the just-inserted entry), padding earlier
                // entries with a placeholder default. Callers must
                // re-emit positions for those frames in a subsequent
                // index pass.
                let mut refs = vec![PositionsRef::default(); self.frame_ids.len()];
                refs[idx] = r;
                self.positions_refs = Some(refs);
            }
            (Some(refs), maybe_ref) => {
                let r = maybe_ref.unwrap_or_default();
                if replace_existing {
                    refs[idx] = r;
                } else {
                    refs.insert(idx, r);
                }
            }
        }
    }

    /// Locate the index of `frame_id` in this list, or `None`.
    #[cfg(test)]
    pub(crate) fn position_of(&self, frame_id: u64) -> Option<usize> {
        self.frame_ids.binary_search(&frame_id).ok()
    }
}

/// In-memory representation of one text-space's canonical state.
///
/// Term lookup has two backings (Exploit 2):
/// - `term_id_by_bytes` (HashMap): populated on the **write** path via
///   `build_flush_output`, where O(1) amortized inserts dominate.
/// - `sorted_terms` (sorted Vec): populated on the **open** path via
///   `rebuild_text_space_state`, where avoiding the per-term HashMap
///   `insert` saves ~30 ms on 200 k-corpus open. Query lookups go through
///   `lookup_term` which prefers the sorted Vec when non-empty.
///
/// Postings storage has **two** parallel representations (Exploit A1):
/// - `postings_by_term` (HashMap → SoA): populated by the **write** path.
///   On open after a commit, this is left empty and queries hit the
///   directory below.
/// - `postings_segments` + `postings_locator`: populated by the **open**
///   fast path. Each segment directory has fixed-size 24-byte entries
///   pointing at varint blob slices; lazy decode happens on the BM25 hot
///   path, ~3 µs per query term.
///
/// Read-time scratch caches for BM25. Populated lazily on the first query
/// and cleared on any mutation that changes `doc_lengths` or the active-row
/// status. Held under `parking_lot::Mutex` so query callers can update them
/// through a shared `&TextSpaceState`. Lock contention is negligible for the
/// single-threaded bench — a typical query holds the lock only long enough
/// to clone the `Arc<Vec<f32>>` doc_normalizer (pointer bump, no copy).
#[derive(Debug, Default)]
pub(crate) struct Bm25QueryCache {
    /// `(n_active, total_len_active)` over `doc_lengths`. Cache is only
    /// used when the catalog has zero tombstoned frames; tombstoned
    /// queries fall through to a fresh scan.
    pub(crate) corpus_stats: Option<(u64, u64)>,
    /// Most-recent doc_normalizer keyed by `(k1, b, avg_dl, n_frames)`.
    /// Held behind `Arc` so the query path clones the pointer, not the
    /// 4×n_frames byte buffer.
    pub(crate) doc_normalizer: Option<DocNormalizerCache>,
    /// Per-doc count-cosine L2 norm: `sqrt(sum_t tf(t,d)^2)`.
    /// Invariant of any query parameter; depends only on doc-side tf
    /// distribution. One entry per index epoch; rebuilt on
    /// `doc_lengths` change via `invalidate_bm25_cache`.
    pub(crate) count_l2_norms: Option<CountL2Cache>,
    /// Per-doc tf-idf cosine L2 norms, keyed by `(tf_mode, n_active)`.
    /// Up to one entry per `(TfMode, n_active.to_bits())` combination
    /// active in the workload. For BEIR (no tombstones) the cache
    /// fills with at most two entries after the first cosine query of
    /// each `tf_mode`.
    pub(crate) tfidf_l2_norms: Vec<TfidfL2Cache>,
    /// Per-term **impact-sorted view** of the posting list:
    /// `(frame_ids, term_freqs)` permuted so that the highest-tf
    /// documents come first (tie-broken by ascending `frame_id`).
    /// Built lazily on first access via
    /// `tfidf::lookup_or_build_impact_posting`. The original
    /// frame_id-sorted form is preserved on disk and accessed via
    /// `lookup_postings_sparse` for binary-search reranks.
    pub(crate) impact_postings: HashMap<TermId, Arc<ImpactPosting>>,
    /// Per-term **frame_id-sorted** decoded posting cache. Shares the
    /// same single-decode pass that builds the impact view, so the
    /// rerank phase can binary-search without paying the cost of a
    /// second varint decode of the segment blob.
    pub(crate) decoded_postings:
        HashMap<TermId, Arc<crate::format::postings::PostingListSoaDecoded>>,
}

/// Posting list reordered by **descending term frequency**, with
/// `frame_id` ascending as the tie-breaker. The vote phase of the
/// impact-vote-rerank scorers reads only the first `channel_k`
/// entries — by construction, those carry the highest-energy
/// contributions for the term.
#[derive(Debug)]
pub(crate) struct ImpactPosting {
    pub(crate) frame_ids: Vec<u64>,
    pub(crate) term_freqs: Vec<u32>,
}

#[derive(Debug)]
pub(crate) struct DocNormalizerCache {
    pub(crate) k1_bits: u32,
    pub(crate) b_bits: u32,
    pub(crate) avg_dl_bits: u32,
    pub(crate) n_frames: usize,
    pub(crate) vec: Arc<Vec<f32>>,
}

#[derive(Debug)]
pub(crate) struct CountL2Cache {
    pub(crate) n_frames: usize,
    pub(crate) vec: Arc<Vec<f32>>,
}

#[derive(Debug)]
pub(crate) struct TfidfL2Cache {
    /// 0 = `TfMode::Raw`, 1 = `TfMode::Log`. Matches the dispatch in
    /// `tfidf::lookup_or_build_tfidf_l2_norms`.
    pub(crate) tf_mode_id: u8,
    pub(crate) n_active_bits: u32,
    pub(crate) n_frames: usize,
    pub(crate) vec: Arc<Vec<f32>>,
}

#[derive(Debug, Default)]
pub(crate) struct TextSpaceState {
    pub(crate) term_id_by_bytes: HashMap<Vec<u8>, TermId>,
    /// Sorted term-bytes index. Each entry is `(blob_idx, start, length,
    /// term_id)` and references into `term_dict_blobs[blob_idx]` —
    /// **zero per-term heap allocation**. Lookup compares the live mmap
    /// payload bytes against the query term. Populated by the open-time
    /// chain replay; never updated by the write path (which keeps the
    /// HashMap fresh instead).
    pub(crate) sorted_terms: Vec<TermSlice>,
    /// Kept-alive term-dictionary segment payloads. `sorted_terms`
    /// references into these `Vec<u8>` allocations so the per-term
    /// `Vec<u8>` heap copy is collapsed to one allocation per segment.
    pub(crate) term_dict_blobs: Vec<Vec<u8>>,
    /// Running df per term_id (sum of df_delta across the postings
    /// chain). **Multi-segment fallback only** — the single-segment
    /// fast path reads df straight from the entry via
    /// `single_segment_dir_index`. Empty when the fast path is active.
    pub(crate) df_by_term: HashMap<TermId, u32>,
    /// Running cf per term_id (sum of (term_freq) across all postings).
    /// Multi-segment fallback (see `df_by_term`).
    pub(crate) cf_by_term: HashMap<TermId, u64>,
    /// term_id → SoA posting list (postings sorted ascending by frame_id).
    /// **Write-side cache only**: populated by `build_flush_output` (live
    /// indexing). Empty on cold open — queries lazy-decode from the
    /// directory below. Tests that drive both paths in-process see this
    /// field populated through the write path.
    pub(crate) postings_by_term: HashMap<TermId, PostingListSoA>,
    /// Loaded postings segments (oldest-first per chain). Each segment
    /// holds its own `entries` directory + retained `blob` bytes; lazy
    /// decode at query time slices into `blob` using `(blob_offset,
    /// blob_len)` from the matching entry.
    pub(crate) postings_segments: Vec<PostingsSegmentDirectory>,
    /// **Single-segment fast path**: when `postings_segments.len() == 1`,
    /// this Vec is indexed by `term_id.0 as usize` and holds the entry
    /// index into `postings_segments[0].entries`. `u32::MAX` means "no
    /// posting list for this term_id in this segment". Populated at
    /// open in O(directory_entries) and replaces the
    /// `postings_locator` HashMap for the bench scenario. Empty in the
    /// multi-segment case.
    pub(crate) single_segment_dir_index: Vec<u32>,
    /// **Multi-segment fallback** for `lookup_postings`. Empty when the
    /// single-segment fast path is active.
    pub(crate) postings_locator: HashMap<TermId, Vec<PostingsLocator>>,
    /// frame_id → docstats (newest wins on chain replay). Used only by cold
    /// paths (segment encoding); the hot retrieval path uses `doc_lengths`
    /// and `doc_unique_terms` for O(1) dense lookup.
    pub(crate) doc_stats: HashMap<FrameId, DocStatsEntry>,
    /// Dense `total_terms` per frame, indexed by `FrameId.0`. Entries that
    /// are 0 indicate the frame is not indexed in this text-space (or
    /// genuinely had zero terms — both are scored as zero anyway).
    pub(crate) doc_lengths: Vec<u32>,
    /// Dense `unique_terms` per frame, indexed by `FrameId.0`. Used by the
    /// Jaccard family for `|d|`.
    pub(crate) doc_unique_terms: Vec<u32>,
    /// Highest `FrameId.0` ever observed in this text-space (sets the dense
    /// buffer length: `doc_lengths.len() == max_frame_id + 1`).
    pub(crate) max_frame_id: u64,
    /// Next term_id to assign. Persisted indirectly via the term-dict chain
    /// (terms in chain dictate the next id at open time).
    pub(crate) next_term_id: u32,
    /// Read-time BM25 scratch caches (corpus_stats, doc_normalizer).
    /// Cleared by `record_doc_stats` and `ensure_frame_capacity` whenever
    /// they observe a change.
    pub(crate) bm25_cache: Mutex<Bm25QueryCache>,
}

impl TextSpaceState {
    /// Ensure dense per-frame buffers cover `frame_id`. Cheap no-op when
    /// `frame_id <= max_frame_id`. Allocates O(growth) on growth.
    pub(crate) fn ensure_frame_capacity(&mut self, frame_id: FrameId) {
        let needed = frame_id.0.saturating_add(1) as usize;
        if needed > self.doc_lengths.len() {
            self.doc_lengths.resize(needed, 0);
            self.doc_unique_terms.resize(needed, 0);
            self.invalidate_bm25_cache();
        }
        if frame_id.0 > self.max_frame_id {
            self.max_frame_id = frame_id.0;
        }
    }

    /// Record a docstats entry into the dense per-frame buffers. Newest-wins
    /// is implicit: each call overwrites the prior value.
    pub(crate) fn record_doc_stats(&mut self, entry: &DocStatsEntry) {
        self.ensure_frame_capacity(entry.frame_id);
        let idx = entry.frame_id.0 as usize;
        self.doc_lengths[idx] = entry.total_terms;
        self.doc_unique_terms[idx] = entry.unique_terms;
        self.invalidate_bm25_cache();
    }

    /// Drop any cached BM25 / cosine scratch. Called by every mutation
    /// that touches `doc_lengths`. Cheap: a single mutex lock plus a
    /// handful of `take`s / `clear`s.
    fn invalidate_bm25_cache(&mut self) {
        let mut g = self.bm25_cache.lock();
        g.corpus_stats = None;
        g.doc_normalizer = None;
        g.count_l2_norms = None;
        g.tfidf_l2_norms.clear();
        g.impact_postings.clear();
        g.decoded_postings.clear();
    }

    /// Resolve `bytes` to its TermId. Tries the sorted Vec first (the
    /// post-open read path), then falls through to the HashMap (covers
    /// post-write deltas appended after open in a long-running RW handle,
    /// where new terms land in the HashMap until the next commit cycle).
    /// Returns `None` if the term is unknown to both.
    ///
    /// Each comparison reads `&term_dict_blobs[blob_idx][start..start+length]`
    /// — the bytes live in the segment payloads kept alive by
    /// `term_dict_blobs`, not in per-term `Vec<u8>` allocations.
    #[inline]
    pub(crate) fn lookup_term(&self, bytes: &[u8]) -> Option<TermId> {
        if !self.sorted_terms.is_empty()
            && let Ok(i) = self.sorted_terms.binary_search_by(|slice| {
                let blob = &self.term_dict_blobs[slice.blob_idx as usize];
                let s = slice.start as usize;
                let e = s + slice.length as usize;
                blob[s..e].cmp(bytes)
            })
        {
            return Some(self.sorted_terms[i].term_id);
        }
        self.term_id_by_bytes.get(bytes).copied()
    }

    /// Resolve `term_id` → document frequency (number of frames the
    /// term appears in). Reads directly from the postings directory
    /// entry in the single-segment fast path; falls back to the
    /// `df_by_term` HashMap for multi-segment files. Returns 0 for
    /// unknown terms.
    #[inline]
    pub(crate) fn df_for(&self, term_id: TermId) -> u32 {
        if !self.single_segment_dir_index.is_empty()
            && let Some(&entry_idx) = self.single_segment_dir_index.get(term_id.0 as usize)
            && entry_idx != NO_POSTING_ENTRY
        {
            return self.postings_segments[0].entries[entry_idx as usize].df;
        }
        self.df_by_term.get(&term_id).copied().unwrap_or(0)
    }

    /// Resolve `term_id` → collection frequency (sum of term-frequency
    /// across all postings). Currently unused on retrieval paths;
    /// retained for symmetry with `df_for` and future scoring needs.
    #[allow(dead_code)]
    #[inline]
    pub(crate) fn cf_for(&self, term_id: TermId) -> u64 {
        if !self.single_segment_dir_index.is_empty()
            && let Some(&entry_idx) = self.single_segment_dir_index.get(term_id.0 as usize)
            && entry_idx != NO_POSTING_ENTRY
        {
            return self.postings_segments[0].entries[entry_idx as usize].cf;
        }
        self.cf_by_term.get(&term_id).copied().unwrap_or(0)
    }

    /// Lazy posting-list lookup (Exploit A1 + Exploit 3).
    ///
    /// Cold open populates only `postings_segments` + `postings_locator`
    /// — no varint decode at open time. This method returns a borrowed
    /// view into the segment blob:
    /// - `Sparse`: varint-decoded SoA arrays (~3 µs per term)
    /// - `Dense` (Roaring fast-path): a `&[u8]` slice into the packed
    ///   bitset. Zero allocation; the scorer iterates set bits.
    ///
    /// Falls back to `postings_by_term` (in-memory write-side cache) so
    /// tests that build state through `build_flush_output` continue to
    /// work without ever touching disk.
    #[inline]
    pub(crate) fn lookup_postings(&self, term_id: TermId) -> Option<LazyPostingList<'_>> {
        // **Single-segment fast path** (Implicit-Offset Postings Directory):
        // when the chain has exactly one root, `single_segment_dir_index`
        // is populated. `term_id.0 as usize` is a direct index into the
        // entry array — no HashMap probe.
        if !self.single_segment_dir_index.is_empty()
            && let Some(&entry_idx) = self.single_segment_dir_index.get(term_id.0 as usize)
            && entry_idx != NO_POSTING_ENTRY
        {
            let seg = &self.postings_segments[0];
            let entry = &seg.entries[entry_idx as usize];
            let blob_slice = &seg.blob[entry.blob_offset as usize
                ..(entry.blob_offset as usize + entry.blob_len as usize)];
            if entry.is_dense {
                return Some(LazyPostingList::Dense {
                    term_id: entry.term_id,
                    df: entry.df,
                    bitset: blob_slice,
                });
            }
            return decode_term_blob_into(entry.term_id, entry.df, blob_slice)
                .ok()
                .map(LazyPostingList::Sparse);
        }
        if let Some(locators) = self.postings_locator.get(&term_id)
            && let Some(loc) = locators.first()
            && locators.len() == 1
        {
            // Single-segment scenario reached through the HashMap (e.g.,
            // tests that drive `build_flush_output` directly without
            // going through the open path).
            let seg = &self.postings_segments[loc.segment_idx as usize];
            let entry = &seg.entries[loc.entry_idx as usize];
            let blob_slice = &seg.blob[entry.blob_offset as usize
                ..(entry.blob_offset as usize + entry.blob_len as usize)];
            if entry.is_dense {
                return Some(LazyPostingList::Dense {
                    term_id: entry.term_id,
                    df: entry.df,
                    bitset: blob_slice,
                });
            }
            return decode_term_blob_into(entry.term_id, entry.df, blob_slice)
                .ok()
                .map(LazyPostingList::Sparse);
        }
        if let Some(locators) = self.postings_locator.get(&term_id) {
            // Multi-segment: decode + merge through the slow path.
            return decode_term_locators(&self.postings_segments, locators)
                .ok()
                .flatten()
                .map(LazyPostingList::Sparse);
        }
        // Write-side cache fallback.
        let p = self.postings_by_term.get(&term_id)?;
        if p.frame_ids.is_empty() {
            return None;
        }
        Some(LazyPostingList::Sparse(
            crate::format::postings::PostingListSoaDecoded {
                term_id,
                df: p.frame_ids.len() as u32,
                frame_ids: p.frame_ids.clone(),
                term_freqs: p.term_freqs.clone(),
                collection_ids: p.collection_ids.clone(),
                field_masks: p.field_masks.clone(),
            },
        ))
    }

    /// Materialize a posting list as the canonical SoA form, expanding
    /// dense bitsets to (frame_id, tf=1) tuples on the fly. Used by
    /// jaccard / TF-IDF retrieval paths that don't yet have a native
    /// bitset scorer; BM25's hot path uses `lookup_postings` directly to
    /// keep the dense fast-path zero-allocation.
    pub(crate) fn lookup_postings_sparse(
        &self,
        term_id: TermId,
    ) -> Option<crate::format::postings::PostingListSoaDecoded> {
        match self.lookup_postings(term_id)? {
            LazyPostingList::Sparse(p) => Some(p),
            LazyPostingList::Dense {
                term_id,
                df,
                bitset,
            } => Some(expand_bitset_to_sparse(term_id, df, bitset)),
        }
    }

    /// Iterate all terms with at least one posting list. Used by the
    /// full-corpus retrieval paths (count_cosine, tfidf_cosine). Lazy:
    /// each yielded list is freshly decoded.
    pub(crate) fn iter_all_postings(
        &self,
    ) -> Box<dyn Iterator<Item = (TermId, crate::format::postings::PostingListSoaDecoded)> + '_>
    {
        // If the write-side cache is populated, prefer it (avoids
        // decoding bytes that are already structured in memory).
        if !self.postings_by_term.is_empty() {
            return Box::new(self.postings_by_term.iter().filter_map(|(tid, p)| {
                if p.frame_ids.is_empty() {
                    None
                } else {
                    Some((
                        *tid,
                        crate::format::postings::PostingListSoaDecoded {
                            term_id: *tid,
                            df: p.frame_ids.len() as u32,
                            frame_ids: p.frame_ids.clone(),
                            term_freqs: p.term_freqs.clone(),
                            collection_ids: p.collection_ids.clone(),
                            field_masks: p.field_masks.clone(),
                        },
                    ))
                }
            }));
        }
        // Single-segment fast path: walk the dense index directly.
        if !self.single_segment_dir_index.is_empty() && !self.postings_segments.is_empty() {
            let n = self.single_segment_dir_index.len();
            return Box::new((0..n).filter_map(move |i| {
                let entry_idx = self.single_segment_dir_index[i];
                if entry_idx == NO_POSTING_ENTRY {
                    return None;
                }
                let tid = TermId(i as u32);
                self.lookup_postings_sparse(tid).map(|d| (tid, d))
            }));
        }
        Box::new(
            self.postings_locator
                .keys()
                .filter_map(|tid| self.lookup_postings_sparse(*tid).map(|d| (*tid, d))),
        )
    }
}

/// Expand a dense bitset into the canonical SoA form with `tf=1` for
/// every set bit. Allocates `df` u64 frame_ids + tf u32 vectors. Used
/// by the TF-IDF / Jaccard retrieval paths that don't have a native
/// bitset scorer; BM25's hot path skips this entirely.
fn expand_bitset_to_sparse(
    term_id: TermId,
    df: u32,
    bitset: &[u8],
) -> crate::format::postings::PostingListSoaDecoded {
    let n = df as usize;
    let mut frame_ids: Vec<u64> = Vec::with_capacity(n);
    let term_freqs: Vec<u32> = vec![1; n];
    let collection_ids: Vec<u32> = vec![1; n];
    let field_masks: Vec<u32> = vec![0x1; n];
    let words = bitset.len() / 8;
    for word_idx in 0..words {
        let mut word =
            u64::from_le_bytes(bitset[word_idx * 8..word_idx * 8 + 8].try_into().unwrap());
        while word != 0 {
            let bit = word.trailing_zeros() as u64;
            frame_ids.push((word_idx as u64) * 64 + bit);
            word &= word - 1;
        }
    }
    crate::format::postings::PostingListSoaDecoded {
        term_id,
        df,
        frame_ids,
        term_freqs,
        collection_ids,
        field_masks,
    }
}

/// Lazy-decode + merge across one term's locators. Returns `Ok(None)` if
/// the term is somehow present in the locator map but every segment slice
/// decodes to an empty list (shouldn't happen for v1 writers).
fn decode_term_locators(
    segments: &[PostingsSegmentDirectory],
    locators: &[PostingsLocator],
) -> Result<Option<crate::format::postings::PostingListSoaDecoded>> {
    if locators.is_empty() {
        return Ok(None);
    }
    // Fast path: single segment per term — the common no-reflush case.
    if locators.len() == 1 {
        let loc = &locators[0];
        let seg = &segments[loc.segment_idx as usize];
        let entry = &seg.entries[loc.entry_idx as usize];
        let blob_slice = &seg.blob
            [entry.blob_offset as usize..(entry.blob_offset as usize + entry.blob_len as usize)];
        return Ok(Some(decode_term_blob_into(
            entry.term_id,
            entry.df,
            blob_slice,
        )?));
    }
    // Multi-segment merge: decode each, append (frame-ascending across
    // segments by commit order; if not, sort once at the end).
    let mut merged: Option<crate::format::postings::PostingListSoaDecoded> = None;
    for loc in locators {
        let seg = &segments[loc.segment_idx as usize];
        let entry = &seg.entries[loc.entry_idx as usize];
        let blob_slice = &seg.blob
            [entry.blob_offset as usize..(entry.blob_offset as usize + entry.blob_len as usize)];
        let one = decode_term_blob_into(entry.term_id, entry.df, blob_slice)?;
        match merged.as_mut() {
            None => merged = Some(one),
            Some(m) => {
                m.frame_ids.extend(one.frame_ids);
                m.term_freqs.extend(one.term_freqs);
                m.collection_ids.extend(one.collection_ids);
                m.field_masks.extend(one.field_masks);
                m.df += one.df;
            }
        }
    }
    if let Some(m) = merged.as_mut() {
        // Restore frame-ascending invariant if commit order didn't already.
        let n = m.frame_ids.len();
        let mut idx: Vec<u32> = (0..n as u32).collect();
        idx.sort_by_key(|&i| m.frame_ids[i as usize]);
        if idx.iter().enumerate().any(|(k, &i)| k as u32 != i) {
            let frame_ids = idx.iter().map(|&i| m.frame_ids[i as usize]).collect();
            let term_freqs = idx.iter().map(|&i| m.term_freqs[i as usize]).collect();
            let collection_ids = idx.iter().map(|&i| m.collection_ids[i as usize]).collect();
            let field_masks = idx.iter().map(|&i| m.field_masks[i as usize]).collect();
            m.frame_ids = frame_ids;
            m.term_freqs = term_freqs;
            m.collection_ids = collection_ids;
            m.field_masks = field_masks;
        }
    }
    Ok(merged)
}

/// One frame's pending tokenization result for a (text_space, frame) pair.
#[derive(Clone, Debug)]
pub(crate) struct PendingFrameIndex {
    pub(crate) collection_id: CollectionId,
    /// term_bytes → tf within this frame (single-field v1).
    ///
    /// `FxHashMap` (Opt #2): the per-frame term-count map is only ever
    /// read by `.keys()` and `.iter()` paths downstream — none of them
    /// rely on the lex-ordering BTreeMap provides. The lex order needed
    /// at commit time is built once in `build_flush_output` via a
    /// `BTreeSet<&[u8]>` populated from all frames' keys, so per-frame
    /// ordering is moot. FxHash's non-cryptographic ~1–2 cycles-per-byte
    /// hash replaces BTreeMap's O(log N) cache-cold pointer-chasing
    /// + byte-wise key comparisons.
    ///
    /// We tested a sort-then-RLE alternative (`Vec<(Vec<u8>, u32)>`
    /// built via `sort_unstable` + linear dedup) and it lost on the
    /// BEIR pilot: at N≈200 tokens/doc the ~1,750 fat-pointer
    /// comparisons (each touching two heap cache lines) cost ~15× more
    /// cycles than FxHash's per-byte hashing. The sort-RLE approach
    /// likely wins at MS MARCO scale where N is large enough that hash
    /// collisions and bucket-array footprint dominate, but for the
    /// per-document BEIR scale the hash table is the right tool.
    pub(crate) term_counts: FxHashMap<Vec<u8>, u32>,
    /// Total token count in this frame (single-field v1: `vec![total]`).
    pub(crate) total_terms: u32,
}

impl PendingFrameIndex {
    pub(crate) fn from_tokens(collection_id: CollectionId, tokens: Vec<Vec<u8>>) -> Self {
        let total_terms = tokens.len() as u32;
        let mut term_counts: FxHashMap<Vec<u8>, u32> =
            FxHashMap::with_capacity_and_hasher(tokens.len(), Default::default());
        for token in tokens {
            *term_counts.entry(token).or_insert(0) += 1;
        }
        Self {
            collection_id,
            term_counts,
            total_terms,
        }
    }
}

/// Walk the per-text-space term-dict, postings, and doc-stats chains and
/// rebuild the in-memory `TextSpaceState`. Called at `open()` time. The
/// `id_allocator` parameter is reserved for forward-compat with future
/// per-text-space id observations.
pub(crate) fn rebuild_text_space_state(
    file: &mut File,
    root: &TextIndexRoot,
    id_allocator: &mut IdAllocator,
) -> Result<TextSpaceState> {
    let prof = std::env::var_os("VALISE_OPEN_PROFILE").is_some();
    let mut state = TextSpaceState::default();

    // term_ids are per-text-space, not global — no IdAllocator observation.
    let _ = id_allocator;

    // 1. Term dictionary chain. **Zero-copy term dict**: each segment's
    // raw payload bytes are kept alive in `state.term_dict_blobs`; the
    // `sorted_terms` index is `(blob_idx, start, length, term_id)` —
    // **no per-term `Vec<u8>` allocation**. Lookup compares the live
    // bytes against the query. For 200 k terms this drops ~13 ms of
    // term_dict_apply time (avg 65 ns/term × 200 k = 13 ms of
    // `term_bytes.to_vec()` allocations) plus the corresponding sort
    // cost on smaller heap traffic.
    let mark = std::time::Instant::now();
    let term_dict_lazy = walk_term_dict_chain_lazy(file, root.term_dictionary_root)?;
    if prof {
        crate::prof_eprintln!(
            "      [text-state] walk_term_dict_chain:     {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    let mark = std::time::Instant::now();
    let term_count: usize = term_dict_lazy.iter().map(|(_, slices)| slices.len()).sum();
    state.sorted_terms.reserve(term_count);
    state.term_dict_blobs.reserve(term_dict_lazy.len());
    for (blob, slices) in term_dict_lazy {
        let blob_idx = state.term_dict_blobs.len() as u32;
        state.term_dict_blobs.push(blob);
        for raw in slices {
            if raw.term_id.0 + 1 > state.next_term_id {
                state.next_term_id = raw.term_id.0 + 1;
            }
            state.sorted_terms.push(TermSlice {
                blob_idx,
                start: raw.start,
                length: raw.length,
                term_id: raw.term_id,
            });
        }
    }
    // Sort by the live byte slices — segments are internally lex-sorted
    // already, this is a k-way merge across (rare) multiple segments.
    let blobs_ptr: *const Vec<Vec<u8>> = &state.term_dict_blobs;
    state.sorted_terms.sort_unstable_by(|a, b| {
        // SAFETY: `state.term_dict_blobs` outlives the sort closure
        // (this function holds `state` by `&mut`). We use a raw pointer
        // to avoid a borrow conflict with `sort_unstable_by`'s `&mut
        // self` on `state.sorted_terms`.
        let blobs = unsafe { &*blobs_ptr };
        let ab = &blobs[a.blob_idx as usize][a.start as usize..(a.start + a.length) as usize];
        let bb = &blobs[b.blob_idx as usize][b.start as usize..(b.start + b.length) as usize];
        ab.cmp(bb)
    });
    if prof {
        crate::prof_eprintln!(
            "      [text-state] term_dict_apply:          {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }

    // 2. Postings chain — **Implicit-Offset Postings Directory** for the
    // single-segment case (the bench scenario): a `Vec<u32>` indexed by
    // `term_id.0` directly into `postings_segments[0].entries`. df and
    // cf are read from the entry inline at query time. Drops the
    // 200 k-element df_by_term + cf_by_term + postings_locator HashMap
    // population (~19 ms) to ~1 ms of presized Vec push.
    let mark = std::time::Instant::now();
    let postings_chain = walk_postings_chain_directory(file, root.postings_root)?;
    if prof {
        crate::prof_eprintln!(
            "      [text-state] walk_postings_chain:      {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    let mark = std::time::Instant::now();
    state.postings_segments = postings_chain;
    if state.postings_segments.len() == 1 {
        // Fast path: build the dense single-segment index. Compute
        // max_term_id, presize, fill.
        let entries = &state.postings_segments[0].entries;
        let max_term_id = entries.iter().map(|e| e.term_id.0).max().unwrap_or(0);
        let mut idx_vec: Vec<u32> = vec![NO_POSTING_ENTRY; (max_term_id as usize) + 1];
        for (entry_idx, entry) in entries.iter().enumerate() {
            idx_vec[entry.term_id.0 as usize] = entry_idx as u32;
        }
        state.single_segment_dir_index = idx_vec;
    } else {
        // Multi-segment fallback: HashMaps as before.
        let total_entries: usize = state
            .postings_segments
            .iter()
            .map(|s| s.entries.len())
            .sum();
        state.postings_locator.reserve(total_entries);
        state.df_by_term.reserve(total_entries);
        state.cf_by_term.reserve(total_entries);
        for (segment_idx, seg) in state.postings_segments.iter().enumerate() {
            for (entry_idx, entry) in seg.entries.iter().enumerate() {
                *state.df_by_term.entry(entry.term_id).or_insert(0) += entry.df;
                *state.cf_by_term.entry(entry.term_id).or_insert(0) += entry.cf;
                state
                    .postings_locator
                    .entry(entry.term_id)
                    .or_default()
                    .push(PostingsLocator {
                        segment_idx: segment_idx as u32,
                        entry_idx: entry_idx as u32,
                    });
            }
        }
    }
    if prof {
        crate::prof_eprintln!(
            "      [text-state] postings_apply:           {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }

    // 3. DocStats chain — newest entry per frame_id wins. Populate both the
    //    cold HashMap and the dense per-frame buffers used by retrieval.
    let mark = std::time::Instant::now();
    let doc_stats_chain = walk_doc_stats_chain(file, root.doc_stats_root)?;
    if prof {
        crate::prof_eprintln!(
            "      [text-state] walk_doc_stats_chain:     {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }
    let mark = std::time::Instant::now();
    for seg in doc_stats_chain {
        for entry in seg.records {
            state.record_doc_stats(&entry);
            state.doc_stats.insert(entry.frame_id, entry);
        }
    }
    if prof {
        crate::prof_eprintln!(
            "      [text-state] doc_stats_apply:          {:.3}s",
            mark.elapsed().as_secs_f64()
        );
    }

    Ok(state)
}

/// One term-bytes reference inside a kept-alive segment payload Vec.
/// Returned alongside the payload by `walk_term_dict_chain_lazy` so the
/// caller can store the bytes once (in `TextSpaceState::term_dict_blobs`)
/// and the indexes by reference.
#[derive(Clone, Copy, Debug)]
pub(crate) struct TermDictRawSlice {
    pub(crate) start: u32,
    pub(crate) length: u32,
    pub(crate) term_id: TermId,
}

/// Walk the term-dictionary chain and return per-segment `(payload_bytes,
/// raw_slices)`. Each `raw_slices[i]` references into the matching
/// `payload_bytes` Vec — the caller is expected to keep the payload
/// alive (e.g., in `TextSpaceState::term_dict_blobs`) and use
/// `(blob_idx, start, length, term_id)` references for lookups.
///
/// Bypasses `decode_term_dictionary_segment` so per-term `Vec<u8>`
/// `to_vec()` allocations don't happen — the dominant ~13 ms cost on
/// 200 k-corpus open.
fn walk_term_dict_chain_lazy(
    file: &mut File,
    head: Option<SegmentRef>,
) -> Result<Vec<(Vec<u8>, Vec<TermDictRawSlice>)>> {
    use crate::format::TextSpaceId;

    const MAGIC: [u8; 4] = *b"VLTD";
    const VERSION: u16 = 1;
    const HEADER_FIXED: usize = 4 + 2 + 4 + 4; // magic + ver + text_space_id + record_count

    let mut newest_first: Vec<(Vec<u8>, Vec<TermDictRawSlice>, Option<SegmentRef>)> = Vec::new();
    let mut next = head;
    while let Some(root) = next {
        let payload = read_segment_payload_typed(file, root, SegmentType::TermDictionary)?;
        if payload.len() < HEADER_FIXED + 1 {
            return Err(crate::error::Error::Format(
                "TermDictionarySegment truncated".into(),
            ));
        }
        if payload[0..4] != MAGIC {
            return Err(crate::error::Error::Format(format!(
                "TermDictionarySegment magic mismatch: got {:?}",
                &payload[0..4]
            )));
        }
        let version = u16::from_le_bytes(payload[4..6].try_into().unwrap());
        if version != VERSION {
            return Err(crate::error::Error::Format(format!(
                "TermDictionarySegment version mismatch: got {version}"
            )));
        }
        let _text_space_id = TextSpaceId(u32::from_le_bytes(payload[6..10].try_into().unwrap()));
        let record_count = u32::from_le_bytes(payload[10..14].try_into().unwrap()) as usize;
        // previous_root: 1-byte flag + (optional 56-byte SegmentRef body)
        let mut pos = HEADER_FIXED;
        let prev_flag = *payload.get(pos).ok_or_else(|| {
            crate::error::Error::Format("TermDictionarySegment missing previous_root flag".into())
        })?;
        pos += 1;
        let previous_root = match prev_flag {
            0 => None,
            1 => {
                let need = 8 + 8 + 8 + 32; // segment_id + offset + length + checksum
                if payload.len() < pos + need {
                    return Err(crate::error::Error::Format(
                        "TermDictionarySegment previous_root truncated".into(),
                    ));
                }
                let segment_id = crate::format::SegmentId(u64::from_le_bytes(
                    payload[pos..pos + 8].try_into().unwrap(),
                ));
                pos += 8;
                let offset = u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap());
                pos += 8;
                let length = u64::from_le_bytes(payload[pos..pos + 8].try_into().unwrap());
                pos += 8;
                let mut checksum = [0u8; 32];
                checksum.copy_from_slice(&payload[pos..pos + 32]);
                pos += 32;
                Some(SegmentRef {
                    segment_id,
                    offset,
                    length,
                    checksum,
                })
            }
            other => {
                return Err(crate::error::Error::Format(format!(
                    "TermDictionarySegment previous_root flag must be 0 or 1, got {other}"
                )));
            }
        };

        let mut slices: Vec<TermDictRawSlice> = Vec::with_capacity(record_count);
        for _ in 0..record_count {
            // [term_id u32][term_bytes_len u32][term_bytes][cf u64][df u32]
            if payload.len() < pos + 8 {
                return Err(crate::error::Error::Format(
                    "TermDictionarySegment record header truncated".into(),
                ));
            }
            let term_id = TermId(u32::from_le_bytes(
                payload[pos..pos + 4].try_into().unwrap(),
            ));
            pos += 4;
            let term_bytes_len =
                u32::from_le_bytes(payload[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            if payload.len() < pos + term_bytes_len + 8 + 4 {
                return Err(crate::error::Error::Format(
                    "TermDictionarySegment record body truncated".into(),
                ));
            }
            let term_start = pos as u32;
            // We don't read the bytes — the caller will slice into the
            // kept-alive payload via (start, length).
            pos += term_bytes_len;
            // Skip cf u64 + df u32 — running totals are derived from the
            // postings chain (spec §12.0.4), so the dictionary copies are
            // ignored on the read path.
            pos += 8 + 4;
            slices.push(TermDictRawSlice {
                start: term_start,
                length: term_bytes_len as u32,
                term_id,
            });
        }
        if pos != payload.len() {
            return Err(crate::error::Error::Format(format!(
                "TermDictionarySegment trailing {} bytes after records",
                payload.len() - pos
            )));
        }

        newest_first.push((payload, slices, previous_root));
        next = previous_root;
    }
    newest_first.reverse();
    Ok(newest_first.into_iter().map(|(p, s, _)| (p, s)).collect())
}

fn walk_postings_chain_directory(
    file: &mut File,
    head: Option<SegmentRef>,
) -> Result<Vec<PostingsSegmentDirectory>> {
    let mut newest_first = Vec::new();
    let mut next = head;
    while let Some(root) = next {
        let payload = read_segment_payload_typed(file, root, SegmentType::Postings)?;
        let seg = decode_postings_segment_directory(&payload)?;
        next = seg.previous_root;
        newest_first.push(seg);
    }
    newest_first.reverse();
    Ok(newest_first)
}

fn walk_doc_stats_chain(file: &mut File, head: Option<SegmentRef>) -> Result<Vec<DocStatsSegment>> {
    let mut newest_first = Vec::new();
    let mut next = head;
    while let Some(root) = next {
        let payload = read_segment_payload_typed(file, root, SegmentType::DocStats)?;
        let seg = decode_doc_stats_segment(&payload)?;
        next = seg.previous_root;
        newest_first.push(seg);
    }
    newest_first.reverse();
    Ok(newest_first)
}

/// Result of flushing one text_space's pending frames at commit time. The
/// `text_space_id` is captured for diagnostics; callers usually already
/// know which text_space they asked to flush.
pub(crate) struct FlushOutput {
    #[allow(dead_code)]
    pub(crate) text_space_id: TextSpaceId,
    pub(crate) term_dict_segment: Vec<u8>,
    pub(crate) term_dict_record_count: u32,
    pub(crate) postings_segment: Vec<u8>,
    pub(crate) postings_record_count: u32,
    pub(crate) doc_stats_segment: Vec<u8>,
    pub(crate) doc_stats_record_count: u32,
}

/// Build the three delta segments for a single text_space's pending batch.
/// Updates `state` in place with the new term ids, postings, and doc stats.
/// Does not write to disk — caller appends + registers the segments.
///
/// `profile` (when supplied) accumulates per-step wall-clock measurements;
/// pass `None` to skip the timing calls entirely.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_flush_output(
    text_space_id: TextSpaceId,
    pending: &mut HashMap<FrameId, PendingFrameIndex>,
    state: &mut TextSpaceState,
    prior_term_dict_root: Option<SegmentRef>,
    prior_postings_root: Option<SegmentRef>,
    prior_doc_stats_root: Option<SegmentRef>,
    profile: Option<&mut crate::file::TextIndexBuildProfile>,
) -> Result<FlushOutput> {
    let mut local = crate::file::TextIndexBuildProfile::default();
    let p: &mut crate::file::TextIndexBuildProfile = match profile {
        Some(p) => p,
        None => &mut local,
    };

    // Frame iteration order is fixed once: ascending frame_id. Posting
    // lists built below inherit that order, so no per-list `sort_by_key`
    // pass is needed at the end.
    let t = std::time::Instant::now();
    let mut ordered_frames: Vec<(FrameId, &PendingFrameIndex)> =
        pending.iter().map(|(fid, frame)| (*fid, frame)).collect();
    ordered_frames.sort_by_key(|(fid, _)| fid.0);

    // Step 1: discover new term bytes (lex-sorted for deterministic id
    // assignment per spec §12.0.2). The legacy code called
    // `state.lookup_term` once per *occurrence* — that's
    // ~50M log-N binary searches for the 500k-row dbpedia bench. We
    // now (a) deduplicate token slices into a single set in one pass,
    // then (b) call `lookup_term` once per *unique* token. For dbpedia
    // that's ~50M -> ~1M `lookup_term` calls, an order-of-magnitude
    // drop.
    //
    // The dedup pass uses rayon to fan out across `ordered_frames`,
    // each worker maintaining a thread-local `HashSet<&[u8]>` and
    // merging via `reduce`. Token bytes are borrowed straight out of
    // `pending` — zero allocation during discovery. `BTreeSet` is
    // reserved for the small `new_term_bytes` output (<= unique
    // tokens) where lex order is part of the API contract.
    use rayon::prelude::*;
    use std::collections::HashSet;
    let all_tokens: HashSet<&[u8]> = ordered_frames
        .par_iter()
        .fold(HashSet::<&[u8]>::new, |mut acc, (_, frame)| {
            for term_bytes in frame.term_counts.keys() {
                acc.insert(term_bytes.as_slice());
            }
            acc
        })
        .reduce(HashSet::<&[u8]>::new, |a, b| {
            // Merge into the larger of the two sets so `extend` does fewer
            // hashes (we hash only the smaller set's elements).
            let (mut larger, smaller) = if a.len() >= b.len() { (a, b) } else { (b, a) };
            larger.extend(smaller);
            larger
        });
    let mut new_term_bytes: BTreeSet<&[u8]> = BTreeSet::new();
    for token in all_tokens {
        if state.lookup_term(token).is_none() {
            new_term_bytes.insert(token);
        }
    }
    p.collect_new_terms += t.elapsed();

    // Step 2: assign new term_ids in lex order.
    let t = std::time::Instant::now();
    let mut new_term_records: Vec<(TermId, Vec<u8>)> = Vec::with_capacity(new_term_bytes.len());
    for token in &new_term_bytes {
        let term_id = TermId(state.next_term_id);
        state.next_term_id = state
            .next_term_id
            .checked_add(1)
            .ok_or_else(|| Error::Format("text-space term_id space exhausted".into()))?;
        let owned: Vec<u8> = token.to_vec();
        state.term_id_by_bytes.insert(owned.clone(), term_id);
        new_term_records.push((term_id, owned));
    }
    p.assign_term_ids += t.elapsed();

    // Step 3: flat-array radix-style inversion. We push every (term_id,
    // frame_id, collection_id, tf) tuple into one pre-sized contiguous
    // Vec, then `sort_unstable_by_key` on (term_id, frame_id) once. This
    // replaces a `HashMap<TermId, Vec<Posting>>::entry().or_default().push()`
    // loop — the per-posting hash + capacity-doubling re-allocations were
    // the dominant cache-miss source. Postings for the same term land
    // contiguously after sort, ready to stream straight into the
    // `PostingList` emit pass below. Total postings count is the sum of
    // per-frame unique terms, computed once for one Vec allocation.
    let t = std::time::Instant::now();
    let total_postings: usize = ordered_frames
        .iter()
        .map(|(_, f)| f.term_counts.len())
        .sum();
    let mut docstats_records: Vec<DocStatsEntry> = Vec::with_capacity(ordered_frames.len());
    let mut flat: Vec<(u32, u64, u32, u32)> = Vec::with_capacity(total_postings);
    for (frame_id, frame) in &ordered_frames {
        let unique_terms = frame.term_counts.len() as u32;
        docstats_records.push(DocStatsEntry {
            frame_id: *frame_id,
            collection_id: frame.collection_id,
            text_space_id,
            field_lengths: vec![frame.total_terms],
            total_terms: frame.total_terms,
            unique_terms,
        });
        for (term_bytes, tf) in &frame.term_counts {
            let term_id = state
                .lookup_term(term_bytes)
                .expect("BUG: term_id missing after assignment")
                .0;
            flat.push((term_id, frame_id.0, frame.collection_id.0, *tf));
        }
    }
    // Sort by (term_id, frame_id). frame_id ascending within a term is
    // required by the on-disk PostingList contract (gap encoding).
    flat.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    p.build_postings += t.elapsed();

    // Step 4 + 5 (fused): single streaming pass over the sorted flat
    // tuples. For each contiguous run of (term_id == X), emit one
    // `PostingList`, accumulate the term_id-keyed running df/cf into
    // `state`, append into `state.postings_by_term`, and (if X is a
    // newly-introduced term in this flush) emit the term-dict record.
    // The streaming pass eliminates the second sort, the second hash
    // map walk, and the per-postings.clone() into PostingList — we move
    // owned Vecs once.
    let t = std::time::Instant::now();
    let new_term_id_to_bytes: HashMap<u32, &[u8]> = new_term_records
        .iter()
        .map(|(tid, b)| (tid.0, b.as_slice()))
        .collect();
    let mut posting_lists: Vec<PostingList> = Vec::with_capacity(new_term_records.len() + 64);
    let mut term_dict_records: Vec<TermDictionaryEntry> =
        Vec::with_capacity(new_term_records.len());
    let mut i = 0usize;
    while i < flat.len() {
        let cur_term = flat[i].0;
        let mut j = i + 1;
        while j < flat.len() && flat[j].0 == cur_term {
            j += 1;
        }
        // Materialize the PostingList for this term.
        let mut postings: Vec<Posting> = Vec::with_capacity(j - i);
        let mut p_cf: u64 = 0;
        for &(_, frame_id, coll_id, tf) in &flat[i..j] {
            postings.push(Posting {
                frame_id: FrameId(frame_id),
                collection_id: CollectionId(coll_id),
                field_mask: 0x1, // single-field v1
                term_freq: tf,
                positions_ref: None,
            });
            p_cf += u64::from(tf);
        }
        let p_df = (j - i) as u32;
        let term_id = TermId(cur_term);

        // Update running state.
        *state.df_by_term.entry(term_id).or_insert(0) += p_df;
        *state.cf_by_term.entry(term_id).or_insert(0) += p_cf;
        let entry = state.postings_by_term.entry(term_id).or_default();
        entry.reserve(postings.len());
        for posting in &postings {
            entry.upsert(posting.clone());
        }

        // Emit term-dict record only for newly-introduced terms.
        if let Some(bytes) = new_term_id_to_bytes.get(&cur_term) {
            term_dict_records.push(TermDictionaryEntry {
                term_id,
                term_bytes: bytes.to_vec(),
                collection_freq: p_cf,
                doc_freq: p_df,
            });
        }

        posting_lists.push(PostingList {
            term_id,
            df: p_df,
            postings,
        });
        i = j;
    }
    // term_dict_records are emitted in term_id (== cur_term) order, which
    // matches the lex order assignment in step 2 — so no extra sort is
    // needed for deterministic segment encoding.
    for entry in &docstats_records {
        state.record_doc_stats(entry);
        state.doc_stats.insert(entry.frame_id, entry.clone());
    }
    p.build_term_dict += t.elapsed();
    p.update_state += std::time::Duration::ZERO;

    // 6. Encode each delta segment.
    let t = std::time::Instant::now();
    let term_dict_segment = encode_term_dictionary_segment(&TermDictionarySegment {
        text_space_id,
        previous_root: prior_term_dict_root,
        records: term_dict_records.clone(),
    })?;
    let postings_segment = encode_postings_segment(&PostingsSegment {
        text_space_id,
        previous_root: prior_postings_root,
        posting_lists: posting_lists.clone(),
    })?;
    let doc_stats_segment = encode_doc_stats_segment(&DocStatsSegment {
        text_space_id,
        previous_root: prior_doc_stats_root,
        records: docstats_records.clone(),
    })?;
    p.encode_segments += t.elapsed();

    pending.clear();

    Ok(FlushOutput {
        text_space_id,
        term_dict_segment,
        term_dict_record_count: term_dict_records.len() as u32,
        postings_segment,
        postings_record_count: posting_lists.len() as u32,
        doc_stats_segment,
        doc_stats_record_count: docstats_records.len() as u32,
    })
}

/// Find or create the `TextIndexRoot` entry for `text_space_id` in `body`,
/// returning a mutable reference.
pub(crate) fn text_index_root_mut(
    body: &mut TocFooterBody,
    text_space_id: TextSpaceId,
) -> &mut TextIndexRoot {
    if let Some(idx) = body
        .text_index_roots
        .iter()
        .position(|r| r.text_space_id == text_space_id)
    {
        &mut body.text_index_roots[idx]
    } else {
        body.text_index_roots.push(TextIndexRoot {
            text_space_id,
            term_dictionary_root: None,
            postings_root: None,
            positions_root: None,
            doc_stats_root: None,
            token_set_roots: Vec::new(),
            minhash_signature_roots: Vec::new(),
        });
        body.text_index_roots.last_mut().unwrap()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::postings::decode_postings_segment;
    use crate::format::term_dictionary::decode_term_dictionary_segment;

    #[test]
    fn pending_frame_index_aggregates_term_counts() {
        let p = PendingFrameIndex::from_tokens(
            CollectionId(1),
            vec![
                b"alpha".to_vec(),
                b"beta".to_vec(),
                b"alpha".to_vec(),
                b"gamma".to_vec(),
                b"alpha".to_vec(),
            ],
        );
        assert_eq!(p.total_terms, 5);
        assert_eq!(p.term_counts.get(&b"alpha"[..]), Some(&3));
        assert_eq!(p.term_counts.get(&b"beta"[..]), Some(&1));
        assert_eq!(p.term_counts.get(&b"gamma"[..]), Some(&1));
        assert_eq!(p.term_counts.len(), 3);
    }

    #[test]
    fn posting_list_soa_keeps_sorted_and_replaces_duplicate_frames() {
        let mut list = PostingListSoA::default();
        let mk = |fid: u64, tf: u32| Posting {
            frame_id: FrameId(fid),
            collection_id: CollectionId(1),
            field_mask: 0x1,
            term_freq: tf,
            positions_ref: None,
        };
        list.upsert(mk(5, 1));
        list.upsert(mk(3, 1));
        list.upsert(mk(10, 1));
        list.upsert(mk(5, 99)); // replaces
        assert_eq!(list.len(), 3);
        assert_eq!(list.frame_ids, vec![3, 5, 10]);
        assert_eq!(list.term_freqs, vec![1, 99, 1]);
        assert_eq!(list.position_of(5), Some(1));
        assert_eq!(list.position_of(7), None);
    }

    #[test]
    fn positions_refs_option_roundtrip() {
        // None path: every posting has `positions_ref = None`. The SoA's
        // `positions_refs` Option must stay None; no allocation.
        let mut list = PostingListSoA::default();
        let mk = |fid: u64, refs: Option<PositionsRef>| Posting {
            frame_id: FrameId(fid),
            collection_id: CollectionId(1),
            field_mask: 0x1,
            term_freq: 1,
            positions_ref: refs,
        };
        for fid in [3u64, 5, 10] {
            list.upsert(mk(fid, None));
        }
        assert_eq!(list.frame_ids, vec![3, 5, 10]);
        assert!(
            list.positions_refs.is_none(),
            "no positions_ref present → vec stays unallocated"
        );

        // Some path: insert one posting with a ref. The SoA materializes a
        // parallel vec and pads earlier entries with the default ref.
        let r = PositionsRef {
            segment_id: crate::format::SegmentId(42),
            ordinal: 7,
        };
        list.upsert(mk(8, Some(r)));
        let refs = list
            .positions_refs
            .as_ref()
            .expect("positions_refs materialized after first Some");
        assert_eq!(refs.len(), list.frame_ids.len());
        let idx = list
            .position_of(8)
            .expect("frame 8 present after Some-upsert");
        assert_eq!(refs[idx], r);
        // The padded entries (frames 3, 5, 10 inserted before the Some)
        // get PositionsRef::default(), not the new ref.
        for fid in [3u64, 5, 10] {
            let i = list.position_of(fid).expect("padded frame present");
            assert_eq!(refs[i], PositionsRef::default());
        }
    }

    #[test]
    fn build_flush_output_assigns_term_ids_lex_first_seen() {
        let mut pending = HashMap::new();
        pending.insert(
            FrameId(1),
            PendingFrameIndex::from_tokens(
                CollectionId(1),
                vec![b"zebra".to_vec(), b"apple".to_vec()],
            ),
        );
        pending.insert(
            FrameId(2),
            PendingFrameIndex::from_tokens(
                CollectionId(1),
                vec![b"beta".to_vec(), b"apple".to_vec()],
            ),
        );

        let mut state = TextSpaceState::default();
        let out = build_flush_output(
            TextSpaceId(1),
            &mut pending,
            &mut state,
            None,
            None,
            None,
            None,
        )
        .expect("flush");

        // 3 unique terms: apple, beta, zebra (lex order). IDs assigned 0, 1, 2.
        assert_eq!(state.term_id_by_bytes.len(), 3);
        assert_eq!(state.term_id_by_bytes[&b"apple"[..]], TermId(0));
        assert_eq!(state.term_id_by_bytes[&b"beta"[..]], TermId(1));
        assert_eq!(state.term_id_by_bytes[&b"zebra"[..]], TermId(2));
        assert_eq!(state.next_term_id, 3);
        assert!(pending.is_empty(), "pending cleared after flush");

        // Decode the term-dict segment and verify the records.
        let dict =
            decode_term_dictionary_segment(&out.term_dict_segment).expect("decode term dict");
        let term_ids: Vec<TermId> = dict.records.iter().map(|r| r.term_id).collect();
        assert_eq!(term_ids, vec![TermId(0), TermId(1), TermId(2)]);

        // apple appears in 2 frames → df_delta = 2; beta and zebra in 1 → 1.
        assert_eq!(dict.records[0].doc_freq, 2);
        assert_eq!(dict.records[1].doc_freq, 1);
        assert_eq!(dict.records[2].doc_freq, 1);
    }

    #[test]
    fn build_flush_output_subsequent_call_only_emits_new_terms() {
        let mut state = TextSpaceState::default();
        let mut pending = HashMap::new();
        pending.insert(
            FrameId(1),
            PendingFrameIndex::from_tokens(CollectionId(1), vec![b"alpha".to_vec()]),
        );
        build_flush_output(
            TextSpaceId(1),
            &mut pending,
            &mut state,
            None,
            None,
            None,
            None,
        )
        .expect("flush 1");

        // Second commit: alpha + new term beta.
        let mut pending2 = HashMap::new();
        pending2.insert(
            FrameId(2),
            PendingFrameIndex::from_tokens(
                CollectionId(1),
                vec![b"alpha".to_vec(), b"beta".to_vec()],
            ),
        );
        let out = build_flush_output(
            TextSpaceId(1),
            &mut pending2,
            &mut state,
            None,
            None,
            None,
            None,
        )
        .expect("flush 2");

        // Term-dict delta should contain ONLY beta (alpha already known).
        let dict = decode_term_dictionary_segment(&out.term_dict_segment).unwrap();
        assert_eq!(dict.records.len(), 1);
        assert_eq!(dict.records[0].term_bytes, b"beta".to_vec());
        assert_eq!(dict.records[0].term_id, TermId(1));

        // Postings delta should contain BOTH alpha (frame 2) and beta (frame 2).
        let postings = decode_postings_segment(&out.postings_segment).unwrap();
        assert_eq!(postings.posting_lists.len(), 2);
        // Sorted by term_id ascending.
        assert_eq!(postings.posting_lists[0].term_id, TermId(0));
        assert_eq!(postings.posting_lists[1].term_id, TermId(1));
    }

    #[test]
    fn text_index_root_mut_creates_or_finds_entry() {
        let mut body = TocFooterBody::default();
        let r1 = text_index_root_mut(&mut body, TextSpaceId(5));
        r1.term_dictionary_root = Some(SegmentRef {
            segment_id: crate::format::SegmentId(1),
            offset: 0,
            length: 0,
            checksum: [0; 32],
        });
        assert_eq!(body.text_index_roots.len(), 1);

        let r2 = text_index_root_mut(&mut body, TextSpaceId(7));
        assert!(r2.term_dictionary_root.is_none());
        assert_eq!(body.text_index_roots.len(), 2);

        // Re-fetch existing.
        let r1_again = text_index_root_mut(&mut body, TextSpaceId(5));
        assert!(r1_again.term_dictionary_root.is_some());
        assert_eq!(body.text_index_roots.len(), 2);
    }
}
