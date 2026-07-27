// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! TF-IDF retrieval, spec §13.2 + §12.0.5 (derivable cosines).
//!
//! v1 dispatches profile-driven TF-IDF on `(tf_mode, idf_mode, norm_mode)`:
//! - `TfMode::Raw` → `tf`
//! - `TfMode::Log` → `1 + ln(tf)` for tf > 0
//! - `IdfMode::Smooth` → `ln(1 + N/df)` (single v1 variant)
//! - `NormMode::None` → score is the raw sum of tf*idf
//! - `NormMode::L2` → divide by per-doc L2 norm of the tf*idf vector
//!
//! Plus profile-free `count_cosine_query` and `tfidf_cosine_query` per
//! §12.0.5 — both compute cosine similarity over the canonical postings
//! without requiring a registered `RetrievalProfileDesc`.
//!
//! Phase 6 (impact-vote-rerank):
//! - The hot dense buffers (`dots`, `scores`, `touched`) are reused
//!   across queries via a thread-local `QueryArena` with a
//!   generation counter — zero `Vec<f32>` allocations and zero page
//!   faults on the hot path.
//! - When `channel_k` is set, the vote phase reads only the first
//!   `channel_k` entries of each query term's posting list, with
//!   entries pre-sorted by descending `term_freq` at decode time
//!   (lazily cached in `Bm25QueryCache::impact_postings`). This
//!   bounds the candidate set to `|q| · channel_k` regardless of
//!   corpus size.
//! - An exact-rerank phase recomputes the true dot product for the
//!   top-rerank candidates via binary search on the original
//!   frame_id-sorted posting, recovering the exact cosine score.

use std::collections::HashMap;
use std::sync::Arc;

use crate::{
    error::Result,
    file::query_arena::with_arena,
    file::text_indexing::{ImpactPosting, TextSpaceState},
    format::{
        TermId,
        catalog::{FrameStub, IdfMode, NormMode, TfMode},
    },
    retrieval::Hit,
};

use super::bm25::{active_corpus_stats, build_tombstone_set};
use super::top_k::select_top_k_from_touched;

pub(crate) fn tfidf_query(
    state: &TextSpaceState,
    frames: &[FrameStub],
    tf_mode: TfMode,
    idf_mode: IdfMode,
    norm_mode: NormMode,
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
    channel_k: Option<usize>,
) -> Result<Vec<Hit>> {
    if query_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let tombstoned = build_tombstone_set(frames);
    let (n_active_u64, _) = active_corpus_stats(state, &tombstoned);
    let n_active = n_active_u64 as f32;
    if n_active == 0.0 {
        return Ok(Vec::new());
    }
    let query_term_ids = resolve_query_term_ids(state, query_tokens);
    if query_term_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Per-term `(idf, df)` is reused by both vote and rerank phases;
    // computing it once up front keeps both passes hot.
    let mut term_meta: Vec<(TermId, f32)> = Vec::with_capacity(query_term_ids.len());
    for term_id in &query_term_ids {
        let df_u32 = state.df_for(*term_id);
        if df_u32 == 0 {
            continue;
        }
        let df = df_u32 as f32;
        let idf = match idf_mode {
            IdfMode::Smooth => (1.0 + n_active / df).ln(),
        };
        term_meta.push((*term_id, idf));
    }
    if term_meta.is_empty() {
        return Ok(Vec::new());
    }

    let read_limits: HashMap<TermId, usize> = match channel_k {
        Some(k) => {
            let weighted: Vec<(TermId, f32, u32)> = term_meta
                .iter()
                .map(|(tid, idf)| (*tid, *idf, state.df_for(*tid)))
                .collect();
            water_fill_budget(&weighted, k.saturating_mul(term_meta.len()))
        }
        None => HashMap::new(),
    };

    let n_frames = state.doc_lengths.len();
    with_arena(|arena| {
        arena.begin_query(n_frames);

        for (term_id, idf) in &term_meta {
            let impact = lookup_or_build_impact_posting(state, *term_id);
            let posting_len = impact.frame_ids.len();
            let n = match channel_k {
                Some(_) => match read_limits.get(term_id) {
                    Some(&limit) => limit.min(posting_len),
                    None => continue,
                },
                None => posting_len,
            };
            if n == 0 {
                continue;
            }
            let frame_ids = &impact.frame_ids[..n];
            let term_freqs = &impact.term_freqs[..n];
            for i in 0..n {
                let fid = frame_ids[i];
                if !tombstoned.is_empty() && tombstoned.contains(&fid) {
                    continue;
                }
                let idx = fid as usize;
                if idx >= n_frames {
                    continue;
                }
                let tf = score_tf(tf_mode, term_freqs[i]);
                arena.accumulate(idx, tf * idf);
            }
        }

        // When channel_k is None, the vote is the exact full-posting
        // sum; rerank reduces to the same arithmetic and is skipped.
        if channel_k.is_none() {
            if matches!(norm_mode, NormMode::L2) {
                let l2 = lookup_or_build_tfidf_l2_norms(state, &tombstoned, n_active, tf_mode);
                normalize_in_place(arena, &l2);
            }
            return Ok(select_top_k_from_touched(
                &arena.touched,
                &arena.scores,
                top_k,
            ));
        }

        // With channel_k bound, the vote covers only the top
        // `|q| · channel_k` docs. Pick a generous rerank pool and
        // recompute the exact sum from the full postings.
        let pool = rerank_pool_size(top_k);
        let candidates = select_top_k_from_touched(&arena.touched, &arena.scores, pool);
        let l2 = if matches!(norm_mode, NormMode::L2) {
            Some(lookup_or_build_tfidf_l2_norms(
                state,
                &tombstoned,
                n_active,
                tf_mode,
            ))
        } else {
            None
        };
        let mut reranked = Vec::with_capacity(candidates.len());
        for c in candidates {
            let exact_sum = exact_tfidf_sum(state, &term_meta, tf_mode, c.frame_id.0);
            let final_score = match &l2 {
                Some(l2) => {
                    let n = l2[c.frame_id.0 as usize];
                    if n > 0.0 { exact_sum / n } else { 0.0 }
                }
                None => exact_sum,
            };
            if final_score > 0.0 {
                reranked.push(Hit {
                    frame_id: c.frame_id,
                    score: final_score,
                });
            }
        }
        sort_and_truncate(&mut reranked, top_k);
        Ok(reranked)
    })
}

pub(crate) fn count_cosine_query(
    state: &TextSpaceState,
    frames: &[FrameStub],
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
    channel_k: Option<usize>,
) -> Result<Vec<Hit>> {
    count_cosine_query_inner(
        state,
        frames,
        query_tokens,
        NormSource::Exact,
        top_k,
        channel_k,
    )
}

/// Approximate variant of [`count_cosine_query`]: uses `sqrt(doc_length)`
/// as the per-doc norm instead of the exact `sqrt(sum_t tf(t,d)^2)`.
/// Lucene's `BM25Similarity` uses the same proxy for length
/// normalization; the trade-off is a small NDCG drift in exchange for
/// eliminating the index-wide scan that the exact norm requires.
pub(crate) fn count_cosine_approx_query(
    state: &TextSpaceState,
    frames: &[FrameStub],
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
    channel_k: Option<usize>,
) -> Result<Vec<Hit>> {
    count_cosine_query_inner(
        state,
        frames,
        query_tokens,
        NormSource::SqrtDocLen,
        top_k,
        channel_k,
    )
}

fn count_cosine_query_inner(
    state: &TextSpaceState,
    frames: &[FrameStub],
    query_tokens: &[Vec<u8>],
    norm_source: NormSource,
    top_k: Option<usize>,
    channel_k: Option<usize>,
) -> Result<Vec<Hit>> {
    if query_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let tombstoned = build_tombstone_set(frames);

    let q_counts = crate::retrieval::resolve_query_terms(state, query_tokens);
    if q_counts.is_empty() {
        return Ok(Vec::new());
    }
    let q_norm: f32 = q_counts
        .iter()
        .map(|&(_, c)| (c as f32) * (c as f32))
        .sum::<f32>()
        .sqrt();
    if q_norm == 0.0 {
        return Ok(Vec::new());
    }

    // Compute per-term `(idf, df)` once. Used by both the water-fill
    // budget allocator and (post-channel_k) the rerank phase. Even
    // though count-cosine's *scoring math* doesn't use idf, we use it
    // here as the natural information-content signal for budget
    // allocation: rare terms get more of the per-query posting-read
    // budget than common ones.
    let (n_active_u64, _) = active_corpus_stats(state, &tombstoned);
    let n_active = n_active_u64 as f32;
    let mut term_meta: Vec<(TermId, f32, u32)> = Vec::with_capacity(q_counts.len());
    for &(term_id, _) in &q_counts {
        let df_u32 = state.df_for(term_id);
        if df_u32 == 0 {
            continue;
        }
        let df = df_u32 as f32;
        let idf = if n_active > 0.0 {
            (1.0 + n_active / df).ln()
        } else {
            0.0
        };
        term_meta.push((term_id, idf, df_u32));
    }
    if term_meta.is_empty() {
        return Ok(Vec::new());
    }

    // Channel_k: per-term cap → global Q·channel_k budget redistributed
    // by idf. With `None` the cap is disabled (read every posting).
    let read_limits: HashMap<TermId, usize> = match channel_k {
        Some(k) => water_fill_budget(&term_meta, k.saturating_mul(term_meta.len())),
        None => HashMap::new(),
    };

    let n_frames = state.doc_lengths.len();
    with_arena(|arena| {
        arena.begin_query(n_frames);

        // Phase 1 — vote. Read each query term's impact-sorted
        // posting (descending by tf) up to its water-filled read
        // limit (or the full posting when channel_k is None) and
        // accumulate `q_count · d_tf` into the arena.
        for &(term_id, q_count) in &q_counts {
            let impact = lookup_or_build_impact_posting(state, term_id);
            let posting_len = impact.frame_ids.len();
            let n = match channel_k {
                Some(_) => match read_limits.get(&term_id) {
                    Some(&limit) => limit.min(posting_len),
                    None => continue, // stopword: zero budget
                },
                None => posting_len,
            };
            if n == 0 {
                continue;
            }
            let frame_ids = &impact.frame_ids[..n];
            let term_freqs = &impact.term_freqs[..n];
            let q_count_f = q_count as f32;
            for i in 0..n {
                let fid = frame_ids[i];
                if !tombstoned.is_empty() && tombstoned.contains(&fid) {
                    continue;
                }
                let idx = fid as usize;
                if idx >= n_frames {
                    continue;
                }
                let d_count = term_freqs[i] as f32;
                arena.accumulate(idx, q_count_f * d_count);
            }
        }

        // No channel_k → vote IS the exact dot product. Normalize in
        // place against the arena's scores (fused, no second buffer).
        if channel_k.is_none() {
            apply_count_cosine_norm(arena, state, &tombstoned, norm_source, q_norm);
            return Ok(select_top_k_from_touched(
                &arena.touched,
                &arena.scores,
                top_k,
            ));
        }

        // Phase 2 — pick rerank candidates by truncated vote score.
        let pool = rerank_pool_size(top_k);
        let candidates = select_top_k_from_touched(&arena.touched, &arena.scores, pool);

        // Phase 3 — recompute the exact dot product for each candidate
        // by binary-searching every query term's original
        // frame_id-sorted posting list. This recovers the term-tf
        // contribution from entries past the channel_k cap that the
        // vote phase skipped.
        let originals: HashMap<TermId, _> = q_counts
            .iter()
            .filter_map(|&(tid, _)| lookup_or_decode_original_posting(state, tid).map(|p| (tid, p)))
            .collect();
        let mut reranked = Vec::with_capacity(candidates.len());
        for c in candidates {
            let mut exact_dot = 0.0_f32;
            for &(term_id, q_count) in &q_counts {
                if let Some(post) = originals.get(&term_id)
                    && let Some(tf) = tf_for_fid(&post.frame_ids, &post.term_freqs, c.frame_id.0)
                {
                    exact_dot += (q_count as f32) * (tf as f32);
                }
            }
            let norm = doc_norm_for_count(state, &tombstoned, norm_source, c.frame_id.0);
            if norm > 0.0 {
                let score = exact_dot / (q_norm * norm);
                if score > 0.0 {
                    reranked.push(Hit {
                        frame_id: c.frame_id,
                        score,
                    });
                }
            }
        }
        sort_and_truncate(&mut reranked, top_k);
        Ok(reranked)
    })
}

/// Selects how per-doc L2 norms are obtained at query time.
#[derive(Clone, Copy, Debug)]
enum NormSource {
    /// `sqrt(sum_t tf(t,d)^2)` (count cosine) or
    /// `sqrt(sum_t (tf_score(tf(t,d)) * idf(t))^2)` (tfidf cosine).
    /// Computed once per epoch via the cache in [`Bm25QueryCache`].
    Exact,
    /// `sqrt(state.doc_lengths[fid])` — Lucene's length proxy. No
    /// upfront cost, but loses the exact per-doc tf-distribution
    /// information.
    SqrtDocLen,
}

pub(crate) fn tfidf_cosine_query(
    state: &TextSpaceState,
    frames: &[FrameStub],
    tf_mode: TfMode,
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
    channel_k: Option<usize>,
) -> Result<Vec<Hit>> {
    tfidf_cosine_query_inner(
        state,
        frames,
        tf_mode,
        query_tokens,
        NormSource::Exact,
        top_k,
        channel_k,
    )
}

/// Approximate variant of [`tfidf_cosine_query`]: replaces the
/// `sqrt(sum_t (tf*idf)^2)` per-doc norm with `sqrt(doc_length)`.
/// Same trade-off as [`count_cosine_approx_query`]: drops the
/// index-wide L2 sweep at the cost of small NDCG drift.
pub(crate) fn tfidf_cosine_approx_query(
    state: &TextSpaceState,
    frames: &[FrameStub],
    tf_mode: TfMode,
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
    channel_k: Option<usize>,
) -> Result<Vec<Hit>> {
    tfidf_cosine_query_inner(
        state,
        frames,
        tf_mode,
        query_tokens,
        NormSource::SqrtDocLen,
        top_k,
        channel_k,
    )
}

fn tfidf_cosine_query_inner(
    state: &TextSpaceState,
    frames: &[FrameStub],
    tf_mode: TfMode,
    query_tokens: &[Vec<u8>],
    norm_source: NormSource,
    top_k: Option<usize>,
    channel_k: Option<usize>,
) -> Result<Vec<Hit>> {
    if query_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let tombstoned = build_tombstone_set(frames);
    let (n_active_u64, _) = active_corpus_stats(state, &tombstoned);
    let n_active = n_active_u64 as f32;
    if n_active == 0.0 {
        return Ok(Vec::new());
    }

    let q_counts = crate::retrieval::resolve_query_terms(state, query_tokens);
    // `(term_id, idf, q_w)` for every query term that has a posting
    // list. Cached up front so the vote and rerank passes both reuse
    // the same metadata.
    let mut term_meta: Vec<(TermId, f32, f32)> = Vec::with_capacity(q_counts.len());
    for &(term_id, q_count) in &q_counts {
        let df_u32 = state.df_for(term_id);
        if df_u32 == 0 {
            continue;
        }
        let df = df_u32 as f32;
        let idf = (1.0 + n_active / df).ln();
        let q_w = score_tf(tf_mode, q_count) * idf;
        term_meta.push((term_id, idf, q_w));
    }
    if term_meta.is_empty() {
        return Ok(Vec::new());
    }
    let q_norm: f32 = term_meta.iter().map(|(_, _, w)| w * w).sum::<f32>().sqrt();
    if q_norm == 0.0 {
        return Ok(Vec::new());
    }

    // Water-fill the global Q·channel_k budget across terms by idf.
    // Rare terms (high idf) get served first; their unspent budget
    // cascades to commoner terms.
    let read_limits: HashMap<TermId, usize> = match channel_k {
        Some(k) => {
            let weighted: Vec<(TermId, f32, u32)> = term_meta
                .iter()
                .map(|(tid, idf, _)| (*tid, *idf, state.df_for(*tid)))
                .collect();
            water_fill_budget(&weighted, k.saturating_mul(term_meta.len()))
        }
        None => HashMap::new(),
    };

    let n_frames = state.doc_lengths.len();
    with_arena(|arena| {
        arena.begin_query(n_frames);

        // Phase 1 — vote on impact-sorted postings up to each term's
        // water-filled budget (or full posting under exact mode).
        for (term_id, idf, q_w) in &term_meta {
            let impact = lookup_or_build_impact_posting(state, *term_id);
            let posting_len = impact.frame_ids.len();
            let n = match channel_k {
                Some(_) => match read_limits.get(term_id) {
                    Some(&limit) => limit.min(posting_len),
                    None => continue, // zero-idf stopword
                },
                None => posting_len,
            };
            if n == 0 {
                continue;
            }
            let frame_ids = &impact.frame_ids[..n];
            let term_freqs = &impact.term_freqs[..n];
            for i in 0..n {
                let fid = frame_ids[i];
                if !tombstoned.is_empty() && tombstoned.contains(&fid) {
                    continue;
                }
                let idx = fid as usize;
                if idx >= n_frames {
                    continue;
                }
                let d_w = score_tf(tf_mode, term_freqs[i]) * idf;
                arena.accumulate(idx, q_w * d_w);
            }
        }

        if channel_k.is_none() {
            apply_tfidf_cosine_norm(
                arena,
                state,
                &tombstoned,
                norm_source,
                q_norm,
                tf_mode,
                n_active,
            );
            return Ok(select_top_k_from_touched(
                &arena.touched,
                &arena.scores,
                top_k,
            ));
        }

        let pool = rerank_pool_size(top_k);
        let candidates = select_top_k_from_touched(&arena.touched, &arena.scores, pool);

        // Phase 3 — exact rerank against original frame_id-sorted postings.
        let originals: HashMap<TermId, _> = term_meta
            .iter()
            .filter_map(|(tid, _, _)| {
                lookup_or_decode_original_posting(state, *tid).map(|p| (*tid, p))
            })
            .collect();
        let mut reranked = Vec::with_capacity(candidates.len());
        for c in candidates {
            let mut exact = 0.0_f32;
            for (term_id, idf, q_w) in &term_meta {
                if let Some(post) = originals.get(term_id)
                    && let Some(tf) = tf_for_fid(&post.frame_ids, &post.term_freqs, c.frame_id.0)
                {
                    let d_w = score_tf(tf_mode, tf) * idf;
                    exact += q_w * d_w;
                }
            }
            let norm = doc_norm_for_tfidf(
                state,
                &tombstoned,
                norm_source,
                tf_mode,
                n_active,
                c.frame_id.0,
            );
            if norm > 0.0 {
                let score = exact / (q_norm * norm);
                if score > 0.0 {
                    reranked.push(Hit {
                        frame_id: c.frame_id,
                        score,
                    });
                }
            }
        }
        sort_and_truncate(&mut reranked, top_k);
        Ok(reranked)
    })
}

/// Default rerank candidate pool: scales with `top_k` so the exact
/// rerank pass has enough room to recover docs whose vote-phase score
/// was depressed by the channel_k truncation. 4× top_k is generous;
/// when `top_k` is `None` the pool is unbounded (returns every voted
/// doc for the rerank pass).
#[inline]
fn rerank_pool_size(top_k: Option<usize>) -> Option<usize> {
    top_k.map(|k| k.saturating_mul(4).max(k))
}

/// IDF-weighted water-filling allocator for the per-query posting
/// read budget.
///
/// The baseline per-term cap `channel_k` is a flat budget: every
/// query term gets the same number of reads regardless of its
/// signal-to-noise. That's catastrophic on dense-relevance datasets
/// where the query has a stopword-heavy tail ("how to learn ___") —
/// the stopwords saturate their cap with low-idf docs and the rare
/// term ("python") gets truncated even when its full posting fits
/// within the global budget.
///
/// This allocator treats `Q × channel_k` as a **global budget** and
/// distributes it proportionally to each term's `idf`, processing
/// terms in descending-idf order. Any term whose full posting list
/// `df` is smaller than its fair share returns the surplus to the
/// pool for the remaining (lower-idf) terms. Stopwords with `idf ≈ 0`
/// receive essentially no budget; rare terms can spend up to their
/// entire posting before the next term is considered.
///
/// Returns a `HashMap<TermId, usize>` of per-term read limits aligned
/// with the original `terms` slice. Skips entries whose limit is
/// zero — the caller iterates the map directly.
pub(crate) fn water_fill_budget(
    terms: &[(TermId, f32 /* idf */, u32 /* df */)],
    total_budget: usize,
) -> HashMap<TermId, usize> {
    let mut limits: HashMap<TermId, usize> = HashMap::with_capacity(terms.len());
    if total_budget == 0 || terms.is_empty() {
        return limits;
    }

    // Sort indices by idf descending so the rarest term claims its
    // share first. Stable tie-break by term_id keeps the allocation
    // deterministic across calls with identical query terms.
    let mut order: Vec<usize> = (0..terms.len()).collect();
    order.sort_unstable_by(|&a, &b| {
        terms[b]
            .1
            .partial_cmp(&terms[a].1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| terms[a].0.0.cmp(&terms[b].0.0))
    });

    let mut remaining_budget = total_budget as f64;
    let mut remaining_idf: f64 = terms.iter().map(|t| t.1.max(0.0) as f64).sum();

    if remaining_idf <= 0.0 {
        // Degenerate: every term is a stopword (idf == 0 because
        // df == n_active). Fall back to an equal split across all
        // terms — at least we won't drop the query.
        let share = total_budget / terms.len().max(1);
        for &i in &order {
            let cap = share.min(terms[i].2 as usize);
            if cap > 0 {
                limits.insert(terms[i].0, cap);
            }
        }
        return limits;
    }

    for &i in &order {
        let (tid, idf, df) = terms[i];
        let idf64 = (idf.max(0.0)) as f64;
        if remaining_idf <= 0.0 || remaining_budget <= 0.0 || idf64 <= 0.0 {
            // Either we're out of budget, or this term has zero idf
            // (stopword). Skip — its `limits` entry stays absent and
            // the vote loop reads nothing from this posting.
            remaining_idf -= idf64;
            continue;
        }
        let fair_share = (idf64 / remaining_idf) * remaining_budget;
        let alloc = (fair_share.round() as usize)
            .min(df as usize)
            .min(remaining_budget as usize);
        if alloc > 0 {
            limits.insert(tid, alloc);
        }
        remaining_budget -= alloc as f64;
        remaining_idf -= idf64;
    }
    limits
}

#[inline]
fn sort_and_truncate(hits: &mut Vec<Hit>, top_k: Option<usize>) {
    hits.sort_unstable_by(|a, b| {
        b.score
            .to_bits()
            .cmp(&a.score.to_bits())
            .then_with(|| a.frame_id.0.cmp(&b.frame_id.0))
    });
    if let Some(k) = top_k {
        hits.truncate(k);
    }
}

/// Binary-search a frame_id-sorted posting for `fid`, returning the
/// term frequency if the doc contains the term. The original posting
/// remains sorted by frame_id; the impact-sorted view is a separate
/// query-time cache.
#[inline]
fn tf_for_fid(frame_ids: &[u64], term_freqs: &[u32], fid: u64) -> Option<u32> {
    match frame_ids.binary_search(&fid) {
        Ok(i) => Some(term_freqs[i]),
        Err(_) => None,
    }
}

/// In-place L2 normalization for `count_cosine` exact-path (channel_k
/// disabled): divides each touched entry of `arena.scores` by the
/// per-doc norm and `q_norm`. Fuses what used to be `dots` and
/// `scores` into a single buffer.
fn apply_count_cosine_norm(
    arena: &mut crate::file::query_arena::QueryArena,
    state: &TextSpaceState,
    tombstoned: &std::collections::HashSet<u64>,
    norm_source: NormSource,
    q_norm: f32,
) {
    match norm_source {
        NormSource::Exact => {
            let l2 = lookup_or_build_count_l2_norms(state, tombstoned);
            for &fid32 in &arena.touched {
                let idx = fid32 as usize;
                let n = l2[idx];
                if n > 0.0 {
                    arena.scores[idx] /= q_norm * n;
                } else {
                    arena.scores[idx] = 0.0;
                }
            }
        }
        NormSource::SqrtDocLen => {
            for &fid32 in &arena.touched {
                let idx = fid32 as usize;
                let dl = state.doc_lengths[idx] as f32;
                if dl > 0.0 {
                    arena.scores[idx] /= q_norm * dl.sqrt();
                } else {
                    arena.scores[idx] = 0.0;
                }
            }
        }
    }
}

/// In-place L2 normalization for `tfidf_cosine` exact-path. Mirror of
/// [`apply_count_cosine_norm`] but parametrized over `(tf_mode,
/// n_active)`.
fn apply_tfidf_cosine_norm(
    arena: &mut crate::file::query_arena::QueryArena,
    state: &TextSpaceState,
    tombstoned: &std::collections::HashSet<u64>,
    norm_source: NormSource,
    q_norm: f32,
    tf_mode: TfMode,
    n_active: f32,
) {
    match norm_source {
        NormSource::Exact => {
            let l2 = lookup_or_build_tfidf_l2_norms(state, tombstoned, n_active, tf_mode);
            for &fid32 in &arena.touched {
                let idx = fid32 as usize;
                let n = l2[idx];
                if n > 0.0 {
                    arena.scores[idx] /= q_norm * n;
                } else {
                    arena.scores[idx] = 0.0;
                }
            }
        }
        NormSource::SqrtDocLen => {
            for &fid32 in &arena.touched {
                let idx = fid32 as usize;
                let dl = state.doc_lengths[idx] as f32;
                if dl > 0.0 {
                    arena.scores[idx] /= q_norm * dl.sqrt();
                } else {
                    arena.scores[idx] = 0.0;
                }
            }
        }
    }
}

/// Per-doc norm lookup for the count-cosine rerank phase. Reads from
/// the epoch-cached L2 vector or computes `sqrt(doc_length)` directly.
#[inline]
fn doc_norm_for_count(
    state: &TextSpaceState,
    tombstoned: &std::collections::HashSet<u64>,
    norm_source: NormSource,
    fid: u64,
) -> f32 {
    let idx = fid as usize;
    match norm_source {
        NormSource::Exact => {
            let l2 = lookup_or_build_count_l2_norms(state, tombstoned);
            l2.get(idx).copied().unwrap_or(0.0)
        }
        NormSource::SqrtDocLen => state
            .doc_lengths
            .get(idx)
            .map(|&dl| (dl as f32).sqrt())
            .unwrap_or(0.0),
    }
}

#[inline]
fn doc_norm_for_tfidf(
    state: &TextSpaceState,
    tombstoned: &std::collections::HashSet<u64>,
    norm_source: NormSource,
    tf_mode: TfMode,
    n_active: f32,
    fid: u64,
) -> f32 {
    let idx = fid as usize;
    match norm_source {
        NormSource::Exact => {
            let l2 = lookup_or_build_tfidf_l2_norms(state, tombstoned, n_active, tf_mode);
            l2.get(idx).copied().unwrap_or(0.0)
        }
        NormSource::SqrtDocLen => state
            .doc_lengths
            .get(idx)
            .map(|&dl| (dl as f32).sqrt())
            .unwrap_or(0.0),
    }
}

/// In-place L2 normalization for the tfidf_query (profile-driven)
/// exact-path. The arena holds the unnormalized tf*idf sums.
fn normalize_in_place(arena: &mut crate::file::query_arena::QueryArena, l2: &[f32]) {
    for &fid32 in &arena.touched {
        let idx = fid32 as usize;
        let n = l2[idx];
        if n > 0.0 {
            arena.scores[idx] /= n;
        } else {
            arena.scores[idx] = 0.0;
        }
    }
}

/// Recompute the exact `sum_t (tf_score(tf) * idf)` for a single
/// candidate frame by binary-searching every query term's original
/// posting. Used by the tfidf_query rerank phase.
fn exact_tfidf_sum(
    state: &TextSpaceState,
    term_meta: &[(TermId, f32)],
    tf_mode: TfMode,
    fid: u64,
) -> f32 {
    let mut sum = 0.0_f32;
    for (term_id, idf) in term_meta {
        let post = match lookup_or_decode_original_posting(state, *term_id) {
            Some(p) => p,
            None => continue,
        };
        if let Some(tf) = tf_for_fid(&post.frame_ids, &post.term_freqs, fid) {
            sum += score_tf(tf_mode, tf) * idf;
        }
    }
    sum
}

/// Build (or return the cached) impact-sorted view of the posting
/// list for `term_id`. Sort order is `(term_freq desc, frame_id asc)`
/// so the highest-energy contributions for the term cluster at the
/// head of the list.
///
/// Cached in `Bm25QueryCache::impact_postings` behind an `Arc` so
/// concurrent queries clone the pointer, not the buffers. The same
/// single-decode pass also caches the frame_id-sorted original under
/// [`Bm25QueryCache::decoded_postings`] so the rerank phase can
/// binary-search without redecoding the segment blob. Invalidated
/// alongside the rest of the BM25 cache when `doc_lengths` changes.
pub(crate) fn lookup_or_build_impact_posting(
    state: &TextSpaceState,
    term_id: TermId,
) -> Arc<ImpactPosting> {
    {
        let g = state.bm25_cache.lock();
        if let Some(p) = g.impact_postings.get(&term_id) {
            return p.clone();
        }
    }
    // Cache miss: decode the original frame_id-sorted posting and
    // permute by descending tf. Cache both views together so the
    // rerank phase can binary-search the frame_id-sorted form without
    // paying a second decode.
    let posting = match state.lookup_postings_sparse(term_id) {
        Some(p) => p,
        None => {
            let empty = Arc::new(ImpactPosting {
                frame_ids: Vec::new(),
                term_freqs: Vec::new(),
            });
            state
                .bm25_cache
                .lock()
                .impact_postings
                .insert(term_id, empty.clone());
            return empty;
        }
    };
    let n = posting.frame_ids.len();
    let mut idx: Vec<u32> = (0..n as u32).collect();
    idx.sort_unstable_by(|&a, &b| {
        let ta = posting.term_freqs[a as usize];
        let tb = posting.term_freqs[b as usize];
        tb.cmp(&ta)
            .then_with(|| posting.frame_ids[a as usize].cmp(&posting.frame_ids[b as usize]))
    });
    let mut frame_ids = Vec::with_capacity(n);
    let mut term_freqs = Vec::with_capacity(n);
    for &i in &idx {
        frame_ids.push(posting.frame_ids[i as usize]);
        term_freqs.push(posting.term_freqs[i as usize]);
    }
    let impact_arc = Arc::new(ImpactPosting {
        frame_ids,
        term_freqs,
    });
    let original_arc = Arc::new(posting);
    let mut g = state.bm25_cache.lock();
    g.impact_postings.insert(term_id, impact_arc.clone());
    g.decoded_postings.insert(term_id, original_arc);
    impact_arc
}

/// Cached lookup for the **frame_id-sorted** decoded posting. Hits
/// the cache populated by [`lookup_or_build_impact_posting`]; on miss,
/// triggers the shared build so subsequent rerank queries pay no
/// decode cost.
pub(crate) fn lookup_or_decode_original_posting(
    state: &TextSpaceState,
    term_id: TermId,
) -> Option<Arc<crate::format::postings::PostingListSoaDecoded>> {
    {
        let g = state.bm25_cache.lock();
        if let Some(p) = g.decoded_postings.get(&term_id) {
            return Some(p.clone());
        }
    }
    let _ = lookup_or_build_impact_posting(state, term_id);
    state
        .bm25_cache
        .lock()
        .decoded_postings
        .get(&term_id)
        .cloned()
}

fn score_tf(mode: TfMode, term_freq: u32) -> f32 {
    let tf = term_freq as f32;
    match mode {
        TfMode::Raw => tf,
        TfMode::Log => {
            if tf > 0.0 {
                1.0 + tf.ln()
            } else {
                0.0
            }
        }
    }
}

fn resolve_query_term_ids(state: &TextSpaceState, query_tokens: &[Vec<u8>]) -> Vec<TermId> {
    crate::retrieval::resolve_query_terms(state, query_tokens)
        .into_iter()
        .map(|(t, _)| t)
        .collect()
}

/// Look up or build the per-doc count-cosine L2 norm vector.
///
/// **Epoch-cached** (Fix #1): the norm is `sqrt(sum_t tf(t,d)^2)`, a
/// property of the doc-side term-frequency distribution alone. It
/// doesn't depend on any query parameter and is invariant between
/// commits. We pay the index-wide scan exactly once per epoch (the
/// first cosine query after a commit warms the cache); every
/// subsequent query is an L1-resident `Arc::clone`.
///
/// The tombstoned path bypasses the cache to keep correctness: with
/// tombstones the norm-of-living-docs changes per the tombstone set,
/// and caching every possible set would blow up the working set. The
/// BEIR benchmark has no tombstones; the cached path is the hot path.
fn lookup_or_build_count_l2_norms(
    state: &TextSpaceState,
    tombstoned: &std::collections::HashSet<u64>,
) -> std::sync::Arc<Vec<f32>> {
    if !tombstoned.is_empty() {
        return std::sync::Arc::new(compute_doc_count_l2_norms_full(state, tombstoned));
    }
    let n_frames = state.doc_lengths.len();
    {
        let g = state.bm25_cache.lock();
        if let Some(c) = g.count_l2_norms.as_ref()
            && c.n_frames == n_frames
        {
            return c.vec.clone();
        }
    }
    let v = std::sync::Arc::new(compute_doc_count_l2_norms_full(state, tombstoned));
    state.bm25_cache.lock().count_l2_norms = Some(crate::file::text_indexing::CountL2Cache {
        n_frames,
        vec: v.clone(),
    });
    v
}

/// Look up or build the per-doc tf-idf cosine L2 norm vector.
///
/// **Epoch-cached** (Fix #1), keyed by `(tf_mode, n_active.to_bits())`.
/// `tf_mode` (Raw vs Log) and `n_active` (which feeds IDF) jointly
/// determine the per-doc weight vector; everything else is doc-side
/// static. The cache holds up to one entry per active key (typically
/// two for the BEIR pipeline: one for Raw, one for Log).
fn lookup_or_build_tfidf_l2_norms(
    state: &TextSpaceState,
    tombstoned: &std::collections::HashSet<u64>,
    n_active: f32,
    tf_mode: TfMode,
) -> std::sync::Arc<Vec<f32>> {
    if !tombstoned.is_empty() {
        return std::sync::Arc::new(compute_doc_tfidf_l2_norms_full(
            state, tombstoned, n_active, tf_mode,
        ));
    }
    let n_frames = state.doc_lengths.len();
    let tf_mode_id: u8 = match tf_mode {
        TfMode::Raw => 0,
        TfMode::Log => 1,
    };
    let n_active_bits = n_active.to_bits();
    {
        let g = state.bm25_cache.lock();
        if let Some(c) = g.tfidf_l2_norms.iter().find(|c| {
            c.tf_mode_id == tf_mode_id && c.n_active_bits == n_active_bits && c.n_frames == n_frames
        }) {
            return c.vec.clone();
        }
    }
    let v = std::sync::Arc::new(compute_doc_tfidf_l2_norms_full(
        state, tombstoned, n_active, tf_mode,
    ));
    state
        .bm25_cache
        .lock()
        .tfidf_l2_norms
        .push(crate::file::text_indexing::TfidfL2Cache {
            tf_mode_id,
            n_active_bits,
            n_frames,
            vec: v.clone(),
        });
    v
}

/// Full-corpus L2 norm computation under raw counts. One pass over
/// every term and every posting. Cost is O(total_postings) ≈ O(corpus
/// tokens). Called once per epoch from
/// [`lookup_or_build_count_l2_norms`].
fn compute_doc_count_l2_norms_full(
    state: &TextSpaceState,
    tombstoned: &std::collections::HashSet<u64>,
) -> Vec<f32> {
    let n_frames = state.doc_lengths.len();
    let mut sum_sq: Vec<f32> = vec![0.0_f32; n_frames];
    for (_, postings) in state.iter_all_postings() {
        let frame_ids = postings.frame_ids.as_slice();
        let term_freqs = postings.term_freqs.as_slice();
        for (i, &fid) in frame_ids.iter().enumerate() {
            let idx = fid as usize;
            if !tombstoned.is_empty() && tombstoned.contains(&fid) {
                continue;
            }
            let tf = term_freqs[i] as f32;
            sum_sq[idx] += tf * tf;
        }
    }
    sum_sq.iter_mut().for_each(|v| *v = v.sqrt());
    sum_sq
}

/// Full-corpus L2 norm computation under tf*idf weighting. Called
/// once per `(tf_mode, n_active)` epoch combination from
/// [`lookup_or_build_tfidf_l2_norms`].
fn compute_doc_tfidf_l2_norms_full(
    state: &TextSpaceState,
    tombstoned: &std::collections::HashSet<u64>,
    n_active: f32,
    tf_mode: TfMode,
) -> Vec<f32> {
    let n_frames = state.doc_lengths.len();
    let mut sum_sq: Vec<f32> = vec![0.0_f32; n_frames];
    for (term_id, postings) in state.iter_all_postings() {
        let df = state.df_for(term_id) as f32;
        if df <= 0.0 {
            continue;
        }
        let idf = (1.0 + n_active / df).ln();
        let frame_ids = postings.frame_ids.as_slice();
        let term_freqs = postings.term_freqs.as_slice();
        for (i, &fid) in frame_ids.iter().enumerate() {
            let idx = fid as usize;
            if !tombstoned.is_empty() && tombstoned.contains(&fid) {
                continue;
            }
            let tf = score_tf(tf_mode, term_freqs[i]);
            let w = tf * idf;
            sum_sq[idx] += w * w;
        }
    }
    sum_sq.iter_mut().for_each(|v| *v = v.sqrt());
    sum_sq
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::text_indexing::PendingFrameIndex;
    use crate::format::FrameId;
    use crate::format::{CollectionId, TextSpaceId, catalog::FrameStatus};

    fn frame(id: u64, status: FrameStatus) -> FrameStub {
        FrameStub {
            frame_id: FrameId(id),
            status,
        }
    }

    fn build_state(corpus: &[(u64, &[&str])]) -> TextSpaceState {
        let mut state = TextSpaceState::default();
        let mut pending = HashMap::new();
        for (frame_id, tokens) in corpus {
            let bytes_tokens: Vec<Vec<u8>> = tokens.iter().map(|t| t.as_bytes().to_vec()).collect();
            pending.insert(
                FrameId(*frame_id),
                PendingFrameIndex::from_tokens(CollectionId(1), bytes_tokens),
            );
        }
        crate::file::text_indexing::build_flush_output(
            TextSpaceId(1),
            &mut pending,
            &mut state,
            None,
            None,
            None,
            None,
        )
        .expect("flush");
        state
    }

    fn tokens(words: &[&str]) -> Vec<Vec<u8>> {
        words.iter().map(|w| w.as_bytes().to_vec()).collect()
    }

    #[test]
    fn raw_tf_sums_idf_weighted_terms() {
        let state = build_state(&[
            (1, &["alpha", "alpha", "beta"]),
            (2, &["beta", "gamma"]),
            (3, &["delta"]),
        ]);
        let frames: Vec<_> = (1..=3).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits = tfidf_query(
            &state,
            &frames,
            TfMode::Raw,
            IdfMode::Smooth,
            NormMode::None,
            &tokens(&["alpha", "beta"]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].frame_id, FrameId(1));
        assert_eq!(hits[1].frame_id, FrameId(2));
    }

    #[test]
    fn log_tf_dampens_high_term_freq() {
        let state = build_state(&[(1, &["alpha"; 10]), (2, &["alpha", "alpha"])]);
        let frames: Vec<_> = (1..=2).map(|i| frame(i, FrameStatus::Active)).collect();
        let raw = tfidf_query(
            &state,
            &frames,
            TfMode::Raw,
            IdfMode::Smooth,
            NormMode::None,
            &tokens(&["alpha"]),
            None,
            None,
        )
        .unwrap();
        let log = tfidf_query(
            &state,
            &frames,
            TfMode::Log,
            IdfMode::Smooth,
            NormMode::None,
            &tokens(&["alpha"]),
            None,
            None,
        )
        .unwrap();
        let raw_ratio = raw[0].score / raw[1].score;
        let log_ratio = log[0].score / log[1].score;
        assert!(raw_ratio > log_ratio);
    }

    #[test]
    fn tfidf_cosine_self_match_is_unit() {
        let state = build_state(&[(1, &["alpha", "beta"])]);
        let frames = vec![frame(1, FrameStatus::Active)];
        let hits = tfidf_cosine_query(
            &state,
            &frames,
            TfMode::Raw,
            &tokens(&["alpha", "beta"]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-4);
    }

    #[test]
    fn count_cosine_self_match_is_unit() {
        let state = build_state(&[(1, &["alpha", "beta", "alpha"])]);
        let frames = vec![frame(1, FrameStatus::Active)];
        let hits = count_cosine_query(
            &state,
            &frames,
            &tokens(&["alpha", "beta", "alpha"]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert!((hits[0].score - 1.0).abs() < 1e-4);
    }

    #[test]
    fn count_cosine_orders_by_overlap() {
        let state = build_state(&[
            (1, &["alpha", "beta"]),
            (2, &["alpha", "gamma"]),
            (3, &["delta"]),
        ]);
        let frames: Vec<_> = (1..=3).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits =
            count_cosine_query(&state, &frames, &tokens(&["alpha", "beta"]), None, None).unwrap();
        let f1 = hits.iter().find(|h| h.frame_id == FrameId(1)).unwrap();
        let f2 = hits.iter().find(|h| h.frame_id == FrameId(2)).unwrap();
        assert!(f1.score > f2.score);
        assert!(!hits.iter().any(|h| h.frame_id == FrameId(3)));
    }

    #[test]
    fn tombstoned_frames_excluded() {
        let state = build_state(&[(1, &["alpha"]), (2, &["alpha"])]);
        let frames = vec![
            frame(1, FrameStatus::Active),
            frame(2, FrameStatus::Tombstoned),
        ];
        for hits in [
            tfidf_query(
                &state,
                &frames,
                TfMode::Raw,
                IdfMode::Smooth,
                NormMode::None,
                &tokens(&["alpha"]),
                None,
                None,
            )
            .unwrap(),
            count_cosine_query(&state, &frames, &tokens(&["alpha"]), None, None).unwrap(),
            tfidf_cosine_query(
                &state,
                &frames,
                TfMode::Raw,
                &tokens(&["alpha"]),
                None,
                None,
            )
            .unwrap(),
        ] {
            assert!(hits.iter().all(|h| h.frame_id != FrameId(2)));
        }
    }

    #[test]
    fn empty_query_yields_no_hits() {
        let state = build_state(&[(1, &["alpha"])]);
        let frames = vec![frame(1, FrameStatus::Active)];
        for hits in [
            tfidf_query(
                &state,
                &frames,
                TfMode::Raw,
                IdfMode::Smooth,
                NormMode::None,
                &[],
                None,
                None,
            )
            .unwrap(),
            count_cosine_query(&state, &frames, &[], None, None).unwrap(),
            tfidf_cosine_query(&state, &frames, TfMode::Raw, &[], None, None).unwrap(),
        ] {
            assert!(hits.is_empty());
        }
    }
}
