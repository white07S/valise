//! BM25 retrieval, spec §13.1.
//!
//! Standard BM25 with `k1`, `b` parameters and Robertson-Sparck-Jones
//! smoothed IDF: `ln((N - df + 0.5) / (df + 0.5) + 1)`. v1 supports the
//! single `IdfVariant::RobertsonSparckJones` — the only variant in the v1
//! retrieval-profile enum.
//!
//! Hot-path layout (Phase 5):
//! - posting lists are SoA (`PostingListSoA`): two parallel slices
//!   `&[u64]` (frame_ids) and `&[u32]` (term_freqs) feed a tight,
//!   auto-vectorizable inner loop.
//! - `state.doc_lengths` is a dense `Vec<u32>` indexed by `FrameId.0`, so
//!   `n_active` / `total_len` / per-doc length lookups are pointer-free
//!   contiguous reads.
//! - per-doc length normalization `k1 * (1 - b + b * dl / avg_dl)` is
//!   precomputed once into `doc_normalizer: Vec<f32>` so the inner loop
//!   collapses to `idf * tf * (k1+1) / (tf + doc_normalizer[fid])`.
//! - tombstoned frames are filtered through a tiny `HashSet` of
//!   tombstoned `FrameId.0` values (the common case is empty).
//! - score accumulation writes into a dense `Vec<f32>` indexed by
//!   `FrameId.0`; sparse collection happens through a `touched` index list,
//!   and final ranking uses a min-heap top-K (no full sort).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::{
    error::{Error, Result},
    file::query_arena::with_arena,
    file::text_indexing::{DocNormalizerCache, TextSpaceState},
    format::{
        TermId,
        catalog::{FrameStatus, FrameStub, IdfVariant},
    },
    retrieval::Hit,
};

use super::tfidf::{
    lookup_or_build_impact_posting, lookup_or_decode_original_posting, water_fill_budget,
};
use super::top_k::select_top_k_from_touched_signed;

/// Default top-K returned by BM25 when no explicit cap is supplied. Bench
/// truncates at 10/100; library callers can pass `top_k = None` to keep the
/// historical "return everything" behavior.
const DEFAULT_TOP_K: usize = 1024;

#[allow(clippy::too_many_arguments)]
pub(crate) fn bm25_query_topk(
    state: &TextSpaceState,
    frames: &[FrameStub],
    any_tombstoned: bool,
    k1: f32,
    b: f32,
    idf_variant: IdfVariant,
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
    channel_k: Option<usize>,
) -> Result<Vec<Hit>> {
    let prepared = match prepare_bm25(state, frames, any_tombstoned, k1, b, query_tokens)? {
        Some(p) => p,
        None => return Ok(Vec::new()),
    };
    let mut term_meta = Vec::with_capacity(prepared.query_term_qfs.len());
    for (tid, qf) in &prepared.query_term_qfs {
        let df = state.df_for(*tid);
        if df == 0 {
            continue;
        }
        let idf = bm25_idf(prepared.n_active_f, df as f32, idf_variant);
        term_meta.push((*tid, idf, df, *qf));
    }
    if term_meta.is_empty() {
        return Ok(Vec::new());
    }
    Ok(execute_bm25(
        state,
        &prepared.tombstoned,
        &prepared.doc_normalizer,
        &term_meta,
        k1,
        top_k,
        channel_k,
    ))
}

/// Shared front matter — input validation, tombstone set, cached corpus
/// stats, doc_normalizer lookup, **per-term query-frequency counts**.
/// Returns `None` when no work to do (empty corpus, empty tokens after
/// analysis, etc).
///
/// `query_term_qfs` is the multiset of analyzed query tokens collapsed
/// to `(term_id, qf)` pairs — `qf` is the number of times the term
/// occurred in the query after analysis. This is what the score sum
/// multiplies by, matching Lucene/Tantivy/BM25S behavior. Long
/// arguments with repeated discourse words (ArguAna, Touche) gain
/// 1–6 pp NDCG@10 from this; short queries are unaffected since `qf`
/// is almost always 1.
struct Bm25Prepared {
    tombstoned: HashSet<u64>,
    n_active_f: f32,
    doc_normalizer: Arc<Vec<f32>>,
    query_term_qfs: Vec<(TermId, u32)>,
}

fn prepare_bm25(
    state: &TextSpaceState,
    frames: &[FrameStub],
    any_tombstoned: bool,
    k1: f32,
    b: f32,
    query_tokens: &[Vec<u8>],
) -> Result<Option<Bm25Prepared>> {
    if query_tokens.is_empty() {
        return Ok(None);
    }
    if k1 < 0.0 || !k1.is_finite() {
        return Err(Error::Format(format!(
            "BM25: k1 must be finite and >= 0, got {k1}"
        )));
    }
    if !(0.0..=1.0).contains(&b) || !b.is_finite() {
        return Err(Error::Format(format!("BM25: b must be in [0, 1], got {b}")));
    }
    let tombstoned = if any_tombstoned {
        build_tombstone_set(frames)
    } else {
        HashSet::new()
    };
    let (n_active, total_len) = if any_tombstoned {
        active_corpus_stats(state, &tombstoned)
    } else if let Some(stats) = state.bm25_cache.lock().corpus_stats {
        stats
    } else {
        let stats = active_corpus_stats(state, &tombstoned);
        state.bm25_cache.lock().corpus_stats = Some(stats);
        stats
    };
    if n_active == 0 {
        return Ok(None);
    }
    let n_active_f = n_active as f32;
    let avg_dl = (total_len as f32) / n_active_f;
    if !avg_dl.is_finite() || avg_dl <= 0.0 {
        return Ok(None);
    }
    // Count query-term frequencies. The `Vec`-with-linear-search
    // approach the dedup path used was O(Q²); a HashMap is O(Q) and
    // wins outright once the query has more than a handful of tokens.
    let query_term_qfs = crate::retrieval::resolve_query_terms(state, query_tokens);
    if query_term_qfs.is_empty() {
        return Ok(None);
    }
    let n_frames = state.doc_lengths.len();
    let doc_normalizer = lookup_or_build_doc_normalizer(state, k1, b, avg_dl, n_frames);
    Ok(Some(Bm25Prepared {
        tombstoned,
        n_active_f,
        doc_normalizer,
        query_term_qfs,
    }))
}

/// Impact-vote-rerank pipeline for BM25 (matches the cosine pipeline in
/// `tfidf.rs`):
///
/// 1. **Vote** — for each query term, read the top entries of its
///    impact-sorted (tf-desc) posting up to its water-filled share of
///    the global `|q|·channel_k` budget, and accumulate
///    `idf·(k1+1)·tf / (tf + doc_normalizer[fid])` into the
///    thread-local `QueryArena`. With `channel_k = None` the full
///    posting is consumed and the vote is exact.
/// 2. **Top-K** — bounded-K heap over `arena.touched`/`arena.scores`.
/// 3. **Rerank** (vote+RR mode only) — recompute the exact BM25 score
///    for the top-`4·top_k` candidates by binary-searching every
///    query term's original frame_id-sorted posting (cached). Recovers
///    score contributions from posting entries past the channel_k cap.
#[allow(clippy::too_many_arguments)]
fn execute_bm25(
    state: &TextSpaceState,
    tombstoned: &HashSet<u64>,
    doc_normalizer: &[f32],
    term_meta: &[(TermId, f32, u32, u32)],
    k1: f32,
    top_k: Option<usize>,
    channel_k: Option<usize>,
) -> Vec<Hit> {
    let k1_plus_1 = k1 + 1.0;
    let any_tombstones = !tombstoned.is_empty();
    let n_frames = state.doc_lengths.len();

    // Water-fill needs only `(term_id, idf, df)`. Project once;
    // the resulting Vec is tiny (one entry per distinct query term).
    let read_limits: HashMap<TermId, usize> = match channel_k {
        Some(k) => {
            let projected: Vec<(TermId, f32, u32)> = term_meta
                .iter()
                .map(|(t, idf, df, _qf)| (*t, *idf, *df))
                .collect();
            water_fill_budget(&projected, k.saturating_mul(term_meta.len()))
        }
        None => HashMap::new(),
    };

    with_arena(|arena| {
        arena.begin_query(n_frames);

        for (tid, idf, _df, qf) in term_meta {
            let impact = lookup_or_build_impact_posting(state, *tid);
            let posting_len = impact.frame_ids.len();
            let n = match channel_k {
                Some(_) => match read_limits.get(tid) {
                    Some(&l) => l.min(posting_len),
                    None => continue,
                },
                None => posting_len,
            };
            if n == 0 {
                continue;
            }
            // qf-weighted term coefficient — the only delta from the
            // dedup variant. Matches Lucene/Tantivy/BM25S.
            let term_weight = (*qf as f32) * idf * k1_plus_1;
            let frame_ids = &impact.frame_ids[..n];
            let term_freqs = &impact.term_freqs[..n];
            for i in 0..n {
                let fid = frame_ids[i];
                if any_tombstones && tombstoned.contains(&fid) {
                    continue;
                }
                let idx = fid as usize;
                if idx >= n_frames {
                    continue;
                }
                let tf = term_freqs[i] as f32;
                let denom = tf + doc_normalizer[idx];
                if denom <= 0.0 {
                    continue;
                }
                arena.accumulate(idx, term_weight * tf / denom);
            }
        }

        // Exact mode: vote is the full BM25 sum, no rerank. Uses the
        // signed top-K because BM25+ with tombstones can produce
        // mixed-sign scores (negative IDF when active corpus shrinks
        // below `df`).
        if channel_k.is_none() {
            let cap = top_k.unwrap_or(DEFAULT_TOP_K);
            if cap == 0 {
                return Vec::new();
            }
            return select_top_k_from_touched_signed(&arena.touched, &arena.scores, Some(cap));
        }

        // Vote+rerank: pick a generous candidate pool from the truncated
        // vote, then recompute the exact BM25 score from the cached
        // frame_id-sorted postings.
        let pool = top_k.map(|k| k.saturating_mul(4).max(k));
        let candidates = select_top_k_from_touched_signed(&arena.touched, &arena.scores, pool);
        let originals: HashMap<TermId, _> = term_meta
            .iter()
            .filter_map(|(t, _, _, _)| {
                lookup_or_decode_original_posting(state, *t).map(|p| (*t, p))
            })
            .collect();
        let mut reranked: Vec<Hit> = Vec::with_capacity(candidates.len());
        for c in candidates {
            let idx = c.frame_id.0 as usize;
            if idx >= n_frames {
                continue;
            }
            let dn = doc_normalizer[idx];
            let mut exact = 0.0_f32;
            for (tid, idf, _df, qf) in term_meta {
                if let Some(post) = originals.get(tid)
                    && let Some(tf) = tf_for_fid(&post.frame_ids, &post.term_freqs, c.frame_id.0)
                {
                    let tf_f = tf as f32;
                    let denom = tf_f + dn;
                    if denom > 0.0 {
                        exact += (*qf as f32) * idf * k1_plus_1 * tf_f / denom;
                    }
                }
            }
            // Keep non-zero (incl. negative) scores: BM25+ with tombstones can
            // produce negative IDF once the active corpus shrinks below a
            // term's `df`. Matches the exact branch's signed-score semantics; a
            // `> 0.0` filter here would drop every survivor of such a term.
            if exact != 0.0 {
                reranked.push(Hit {
                    frame_id: c.frame_id,
                    score: exact,
                });
            }
        }
        reranked.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.frame_id.0.cmp(&b.frame_id.0))
        });
        if let Some(k) = top_k {
            reranked.truncate(k);
        }
        reranked
    })
}

#[inline]
fn tf_for_fid(frame_ids: &[u64], term_freqs: &[u32], fid: u64) -> Option<u32> {
    match frame_ids.binary_search(&fid) {
        Ok(i) => Some(term_freqs[i]),
        Err(_) => None,
    }
}

fn bm25_idf(n: f32, df: f32, variant: IdfVariant) -> f32 {
    match variant {
        // Robertson-Sparck-Jones smoothed: ln((N - df + 0.5) / (df + 0.5) + 1)
        // The "+1" yields a non-negative IDF for any df ≤ N (BM25+).
        IdfVariant::RobertsonSparckJones => ((n - df + 0.5) / (df + 0.5) + 1.0).ln(),
    }
}

/// Build the small set of Tombstoned frame ids. The common case is empty,
/// in which case the per-posting branch is essentially free.
pub(crate) fn build_tombstone_set(frames: &[FrameStub]) -> HashSet<u64> {
    let mut set = HashSet::new();
    for f in frames {
        if f.status == FrameStatus::Tombstoned {
            set.insert(f.frame_id.0);
        }
    }
    set
}

/// Sum `n_active` and `total_len_active` over `state.doc_lengths`, skipping
/// entries listed in `tombstoned`. A tight contiguous loop — auto-vectorized
/// by LLVM — over `<= max_frame_id + 1` u32 words.
pub(crate) fn active_corpus_stats(state: &TextSpaceState, tombstoned: &HashSet<u64>) -> (u64, u64) {
    let mut n_active = 0u64;
    let mut total_len = 0u64;
    for (i, &dl) in state.doc_lengths.iter().enumerate() {
        if dl == 0 {
            continue;
        }
        if !tombstoned.is_empty() && tombstoned.contains(&(i as u64)) {
            continue;
        }
        n_active += 1;
        total_len += u64::from(dl);
    }
    (n_active, total_len)
}

/// `doc_normalizer[fid] = k1 * (1 - b + b * dl / avg_dl)`. Single tight
/// loop, vectorizable, dense output indexed by `FrameId.0`.
fn build_doc_normalizer(doc_lengths: &[u32], k1: f32, b: f32, avg_dl: f32) -> Vec<f32> {
    let inv_avg_dl = 1.0 / avg_dl;
    let one_minus_b = 1.0 - b;
    let mut out = vec![0.0_f32; doc_lengths.len()];
    for (dst, &dl) in out.iter_mut().zip(doc_lengths.iter()) {
        if dl == 0 {
            continue;
        }
        let dl_f = dl as f32;
        *dst = k1 * (one_minus_b + b * dl_f * inv_avg_dl);
    }
    out
}

/// Look up a previously-built doc_normalizer by `(k1, b, avg_dl, n_frames)`
/// or build one and stash it back. Stored as `Arc<Vec<f32>>` so the
/// query path clones a pointer (8-byte refcount bump), not the
/// 4×n_frames byte buffer. Cleared by `TextSpaceState::record_doc_stats`
/// and `ensure_frame_capacity` whenever they touch `doc_lengths`.
fn lookup_or_build_doc_normalizer(
    state: &TextSpaceState,
    k1: f32,
    b: f32,
    avg_dl: f32,
    n_frames: usize,
) -> Arc<Vec<f32>> {
    let k1_bits = k1.to_bits();
    let b_bits = b.to_bits();
    let avg_dl_bits = avg_dl.to_bits();
    {
        let g = state.bm25_cache.lock();
        if let Some(c) = g.doc_normalizer.as_ref()
            && c.k1_bits == k1_bits
            && c.b_bits == b_bits
            && c.avg_dl_bits == avg_dl_bits
            && c.n_frames == n_frames
        {
            return c.vec.clone();
        }
    }
    let v = Arc::new(build_doc_normalizer(&state.doc_lengths, k1, b, avg_dl));
    state.bm25_cache.lock().doc_normalizer = Some(DocNormalizerCache {
        k1_bits,
        b_bits,
        avg_dl_bits,
        n_frames,
        vec: v.clone(),
    });
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::text_indexing::PendingFrameIndex;
    use crate::format::FrameId;
    use crate::format::{CollectionId, TextSpaceId};
    use std::collections::HashMap;

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
    fn empty_query_yields_no_hits() {
        let state = build_state(&[(1, &["alpha", "beta"]), (2, &["gamma"])]);
        let frames = vec![frame(1, FrameStatus::Active), frame(2, FrameStatus::Active)];
        let hits = bm25_query_topk(
            &state,
            &frames,
            false,
            1.2,
            0.75,
            IdfVariant::RobertsonSparckJones,
            &[],
            None,
            None,
        )
        .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn unknown_query_term_yields_no_hits() {
        let state = build_state(&[(1, &["alpha"]), (2, &["beta"])]);
        let frames = vec![frame(1, FrameStatus::Active), frame(2, FrameStatus::Active)];
        let hits = bm25_query_topk(
            &state,
            &frames,
            false,
            1.2,
            0.75,
            IdfVariant::RobertsonSparckJones,
            &tokens(&["zeta"]),
            None,
            None,
        )
        .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn single_term_query_orders_by_tf() {
        let state = build_state(&[
            (1, &["alpha", "alpha", "beta"]),
            (2, &["alpha", "gamma", "delta", "epsilon"]),
            (3, &["kappa"]),
        ]);
        let frames: Vec<_> = (1..=3).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits = bm25_query_topk(
            &state,
            &frames,
            false,
            1.2,
            0.75,
            IdfVariant::RobertsonSparckJones,
            &tokens(&["alpha"]),
            None,
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].frame_id, FrameId(1));
        assert_eq!(hits[1].frame_id, FrameId(2));
        assert!(hits[0].score > hits[1].score);
    }

    #[test]
    fn tombstoned_frames_excluded() {
        let state = build_state(&[(1, &["alpha", "beta"]), (2, &["alpha"]), (3, &["alpha"])]);
        let frames = vec![
            frame(1, FrameStatus::Active),
            frame(2, FrameStatus::Tombstoned),
            frame(3, FrameStatus::Active),
        ];
        let hits = bm25_query_topk(
            &state,
            &frames,
            true,
            1.2,
            0.75,
            IdfVariant::RobertsonSparckJones,
            &tokens(&["alpha"]),
            None,
            None,
        )
        .unwrap();
        let ids: Vec<u64> = hits.iter().map(|h| h.frame_id.0).collect();
        assert_eq!(ids, vec![1, 3]);
    }

    #[test]
    fn multi_term_query_sums_scores_and_dominates_when_all_match() {
        let state = build_state(&[
            (1, &["alpha", "beta"]),
            (2, &["alpha"]),
            (3, &["beta"]),
            (4, &["gamma"]),
        ]);
        let frames: Vec<_> = (1..=4).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits = bm25_query_topk(
            &state,
            &frames,
            false,
            1.2,
            0.75,
            IdfVariant::RobertsonSparckJones,
            &tokens(&["alpha", "beta"]),
            None,
            None,
        )
        .unwrap();
        assert!(hits.iter().any(|h| h.frame_id == FrameId(1)));
        let f1_score = hits
            .iter()
            .find(|h| h.frame_id == FrameId(1))
            .unwrap()
            .score;
        for h in &hits {
            if h.frame_id != FrameId(1) {
                assert!(f1_score > h.score);
            }
        }
        assert!(!hits.iter().any(|h| h.frame_id == FrameId(4)));
    }

    #[test]
    fn rejects_invalid_b() {
        let state = TextSpaceState::default();
        let frames: Vec<FrameStub> = Vec::new();
        let err = bm25_query_topk(
            &state,
            &frames,
            false,
            1.2,
            1.5,
            IdfVariant::RobertsonSparckJones,
            &tokens(&["alpha"]),
            None,
            None,
        )
        .expect_err("b > 1");
        assert!(err.to_string().contains("b must be"));
    }

    #[test]
    fn rejects_invalid_k1() {
        let state = TextSpaceState::default();
        let frames: Vec<FrameStub> = Vec::new();
        let err = bm25_query_topk(
            &state,
            &frames,
            false,
            -1.0,
            0.5,
            IdfVariant::RobertsonSparckJones,
            &tokens(&["alpha"]),
            None,
            None,
        )
        .expect_err("negative k1");
        assert!(err.to_string().contains("k1 must be"));
    }

    #[test]
    fn empty_corpus_yields_no_hits() {
        let state = TextSpaceState::default();
        let frames: Vec<FrameStub> = Vec::new();
        let hits = bm25_query_topk(
            &state,
            &frames,
            false,
            1.2,
            0.75,
            IdfVariant::RobertsonSparckJones,
            &tokens(&["alpha"]),
            None,
            None,
        )
        .unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn top_k_caps_results_and_keeps_highest_scoring() {
        // 5 docs all containing "alpha" but with varying tf — top-2 should
        // return the two with highest tf.
        let state = build_state(&[
            (1, &["alpha"]),
            (2, &["alpha", "alpha", "alpha", "alpha", "alpha"]),
            (3, &["alpha", "alpha"]),
            (4, &["alpha", "alpha", "alpha"]),
            (5, &["alpha", "alpha", "alpha", "alpha"]),
        ]);
        let frames: Vec<_> = (1..=5).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits = bm25_query_topk(
            &state,
            &frames,
            false,
            1.2,
            0.75,
            IdfVariant::RobertsonSparckJones,
            &tokens(&["alpha"]),
            Some(2),
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), 2);
        // Highest tf wins (frame 2 with tf=5, frame 5 with tf=4).
        assert_eq!(hits[0].frame_id, FrameId(2));
        assert_eq!(hits[1].frame_id, FrameId(5));
    }
}
