//! Bounded-K top-K extraction over a `touched`/`scores` pair.
//!
//! Three optimizations bundled here, all from the BEIR latency
//! analysis on TF-IDF / cosine / Jaccard-family scorers:
//!
//! **1. Quickselect-class top-K** (avoid the O(N log N) sort over the
//! whole touched set). When `top_k = Some(k)` and `touched.len() > k`,
//! we maintain a bounded `BinaryHeap` of size `k` and only push when
//! a new candidate beats the current worst. Final output is a
//! O(K log K) sort over the K survivors. For TF-IDF on Touche where
//! `touched ≈ 90 k` and `k = 100`, this collapses ~90 k × log(90 k) ≈
//! 1.5 M comparisons down to ~90 k + 100 × log(100) ≈ 90 700.
//!
//! **2. IEEE-754 bit-cast comparison.** Scores produced by every Valise
//! text retriever are non-negative finite f32. For non-negative
//! floats, `f32::to_bits()` preserves total order — comparing the
//! resulting `u32` is a single hardware-native integer compare that
//! the branch predictor handles trivially, replacing the
//! NaN/Inf-handling branches of `partial_cmp`. (NaN safety is
//! enforced upstream: BM25 / TF-IDF / Jaccard formulas only produce
//! NaN if `dl == 0` or `df == 0`, both of which we skip.)
//!
//! **3. L1-resident state.** The heap holds at most `K + 1` 16-byte
//! entries (`HeapEntry` = `u32 score_bits + u64 frame_id + pad`,
//! 16 bytes). For `K = 100` that's 1.6 KB — comfortably L1. The
//! 12 MB-per-query `Vec<Hit>` that the old `collect_hits` allocated
//! on heavy queries is gone entirely.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::format::FrameId;

use super::Hit;

/// One slot in the bounded top-K heap.
///
/// Stores `score.to_bits()` instead of `f32` so the heap's `Ord` impl
/// is a pure integer compare. For non-negative finite floats this is
/// order-preserving; the caller guarantees that constraint before
/// pushing.
#[derive(Clone, Copy, PartialEq, Eq)]
struct HeapEntry {
    score_bits: u32,
    frame_id: u64,
}

impl Ord for HeapEntry {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        // `BinaryHeap` is a max-heap; we use it as a min-heap by
        // putting the WORST candidate at the top (so it's the one
        // popped to make room for a better one).
        //
        // "Worst" ranking semantics (matching the final sort below):
        // - Lower score = worse (lower `to_bits` for non-negative f32).
        // - On a score tie, higher `frame_id` = worse (since we
        //   tie-break by ascending frame_id in the output).
        //
        // So `self.cmp(other) = Greater` iff self is worse than other.
        other
            .score_bits
            .cmp(&self.score_bits)
            .then_with(|| self.frame_id.cmp(&other.frame_id))
    }
}

impl PartialOrd for HeapEntry {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Select up to `top_k` hits from `touched` with the highest `scores`.
///
/// `top_k = None` means "return everything with score > 0", sorted.
/// `top_k = Some(k)` caps the result at `k` entries; the heap-based
/// path activates when `touched.len() > k`.
///
/// Final order is descending by score, ascending by `frame_id` on
/// ties. Same contract as the prior `collect_hits` helpers in the
/// retrieval modules.
pub(crate) fn select_top_k_from_touched(
    touched: &[u32],
    scores: &[f32],
    top_k: Option<usize>,
) -> Vec<Hit> {
    let cap = top_k.unwrap_or(usize::MAX);
    if cap == 0 || touched.is_empty() {
        return Vec::new();
    }

    // Path A: result will be smaller than the cap — just sort.
    if cap >= touched.len() {
        let mut hits: Vec<Hit> = touched
            .iter()
            .filter_map(|&fid32| {
                let s = scores[fid32 as usize];
                if s == 0.0 {
                    None
                } else {
                    Some(Hit {
                        frame_id: FrameId(fid32 as u64),
                        score: s,
                    })
                }
            })
            .collect();
        sort_hits_descending(&mut hits);
        return hits;
    }

    // Path B: bounded heap of size cap. Heap stores u32 score bits to
    // make the comparison branch-free.
    let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(cap + 1);
    for &fid32 in touched {
        let s = scores[fid32 as usize];
        if s <= 0.0 {
            continue;
        }
        let entry = HeapEntry {
            score_bits: s.to_bits(),
            frame_id: fid32 as u64,
        };
        if heap.len() < cap {
            heap.push(entry);
        } else if let Some(worst) = heap.peek()
            && entry.cmp(worst) == Ordering::Less
        {
            // `entry` is better than the heap's worst — swap in.
            heap.pop();
            heap.push(entry);
        }
    }

    let mut hits: Vec<Hit> = heap
        .into_iter()
        .map(|e| Hit {
            frame_id: FrameId(e.frame_id),
            score: f32::from_bits(e.score_bits),
        })
        .collect();
    sort_hits_descending(&mut hits);
    hits
}

/// Signed-score variant of [`select_top_k_from_touched`].
///
/// Used by BM25+: with tombstones the active corpus can be smaller
/// than `df`, which produces a negative `idf` and consequently
/// negative scores. The default helper relies on `f32::to_bits`
/// ordering (only correct for non-negative finite floats); this
/// variant falls back to `partial_cmp` so mixed-sign scores rank
/// correctly. Slightly slower per comparison but only matters when
/// the touched set is large.
pub(crate) fn select_top_k_from_touched_signed(
    touched: &[u32],
    scores: &[f32],
    top_k: Option<usize>,
) -> Vec<Hit> {
    let cap = top_k.unwrap_or(usize::MAX);
    if cap == 0 || touched.is_empty() {
        return Vec::new();
    }
    if cap >= touched.len() {
        let mut hits: Vec<Hit> = touched
            .iter()
            .filter_map(|&fid32| {
                let s = scores[fid32 as usize];
                if s == 0.0 {
                    None
                } else {
                    Some(Hit {
                        frame_id: FrameId(fid32 as u64),
                        score: s,
                    })
                }
            })
            .collect();
        hits.sort_unstable_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.frame_id.0.cmp(&b.frame_id.0))
        });
        return hits;
    }
    // Bounded heap. Same eviction strategy as the unsigned variant
    // but the inner ordering uses `partial_cmp` so mixed-sign scores
    // rank correctly. `(score, frame_id)` lives on the stack at 12 B
    // per slot — for K=100 that's 1.2 KB, L1-resident.
    #[derive(Copy, Clone)]
    struct Entry {
        score: f32,
        frame_id: u64,
    }
    impl PartialEq for Entry {
        fn eq(&self, other: &Self) -> bool {
            self.score == other.score && self.frame_id == other.frame_id
        }
    }
    impl Eq for Entry {}
    impl Ord for Entry {
        fn cmp(&self, other: &Self) -> Ordering {
            other
                .score
                .partial_cmp(&self.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| self.frame_id.cmp(&other.frame_id))
        }
    }
    impl PartialOrd for Entry {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }
    let mut heap: BinaryHeap<Entry> = BinaryHeap::with_capacity(cap + 1);
    for &fid32 in touched {
        let s = scores[fid32 as usize];
        if s == 0.0 {
            continue;
        }
        let entry = Entry {
            score: s,
            frame_id: fid32 as u64,
        };
        if heap.len() < cap {
            heap.push(entry);
        } else if let Some(worst) = heap.peek()
            && entry.cmp(worst) == Ordering::Less
        {
            heap.pop();
            heap.push(entry);
        }
    }
    let mut hits: Vec<Hit> = heap
        .into_iter()
        .map(|e| Hit {
            frame_id: FrameId(e.frame_id),
            score: e.score,
        })
        .collect();
    hits.sort_unstable_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(Ordering::Equal)
            .then_with(|| a.frame_id.0.cmp(&b.frame_id.0))
    });
    hits
}

#[inline]
fn sort_hits_descending(hits: &mut [Hit]) {
    hits.sort_unstable_by(|a, b| {
        // Same bit-cast trick on the final pass for consistency with
        // the heap's ordering. `score.to_bits()` is order-preserving
        // for non-negative f32.
        b.score
            .to_bits()
            .cmp(&a.score.to_bits())
            .then_with(|| a.frame_id.0.cmp(&b.frame_id.0))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn small_touched_returns_all_sorted() {
        let scores = vec![0.0, 0.3, 0.8, 0.1, 0.5];
        let touched = vec![1, 2, 3, 4];
        let hits = select_top_k_from_touched(&touched, &scores, Some(10));
        let expected: Vec<u64> = vec![2, 4, 1, 3];
        let got: Vec<u64> = hits.iter().map(|h| h.frame_id.0).collect();
        assert_eq!(got, expected);
    }

    #[test]
    fn top_k_caps_result() {
        let mut scores = vec![0.0_f32; 1000];
        let mut touched: Vec<u32> = Vec::new();
        for i in 1..1000 {
            scores[i] = (i as f32) * 0.001;
            touched.push(i as u32);
        }
        let hits = select_top_k_from_touched(&touched, &scores, Some(5));
        assert_eq!(hits.len(), 5);
        // Should be 999, 998, 997, 996, 995 (highest scores).
        let got: Vec<u64> = hits.iter().map(|h| h.frame_id.0).collect();
        assert_eq!(got, vec![999, 998, 997, 996, 995]);
    }

    #[test]
    fn ties_broken_by_lowest_frame_id() {
        let scores = vec![0.0, 0.5, 0.5, 0.5, 0.5];
        let touched = vec![4, 2, 1, 3];
        let hits = select_top_k_from_touched(&touched, &scores, Some(2));
        let got: Vec<u64> = hits.iter().map(|h| h.frame_id.0).collect();
        assert_eq!(got, vec![1, 2]);
    }

    #[test]
    fn zero_scores_skipped() {
        let scores = vec![0.0, 0.0, 0.5, 0.0];
        let touched = vec![1, 2, 3];
        let hits = select_top_k_from_touched(&touched, &scores, Some(10));
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].frame_id, FrameId(2));
    }

    #[test]
    fn none_top_k_returns_everything() {
        let scores = vec![0.0, 0.3, 0.5, 0.1];
        let touched = vec![1, 2, 3];
        let hits = select_top_k_from_touched(&touched, &scores, None);
        assert_eq!(hits.len(), 3);
        let got: Vec<u64> = hits.iter().map(|h| h.frame_id.0).collect();
        assert_eq!(got, vec![2, 1, 3]);
    }
}
