//! Exact term-Jaccard retrieval and derivable set-similarity variants,
//! spec §13.3 + §12.0.5.
//!
//! All four functions take the same shape: per-doc score derived from
//! `|q ∩ d|`, `|q|`, and `|d|`, where `|d| = docstats.unique_terms[d]` and
//! `|q ∩ d|` is the count of distinct query term_ids whose posting list
//! contains `d`.
//!
//! - `jaccard`     = `|q ∩ d| / (|q| + |d| − |q ∩ d|)`
//! - `dice`        = `2 |q ∩ d| / (|q| + |d|)`
//! - `overlap`     = `|q ∩ d| / min(|q|, |d|)`
//! - `containment` = `|q ∩ d| / |q|`
//!
//! Tombstoned frames are filtered. v1 only supports `TokenSource::Terms`;
//! shingle-Jaccard requires `TokenSetSegment` (v2).

use crate::{
    error::{Error, Result},
    file::text_indexing::TextSpaceState,
    format::{
        TermId,
        catalog::{FrameStub, TokenSource},
    },
    retrieval::Hit,
};

use super::bm25::build_tombstone_set;
use super::top_k::select_top_k_from_touched;

pub(crate) fn jaccard_query(
    state: &TextSpaceState,
    frames: &[FrameStub],
    token_source: TokenSource,
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
) -> Result<Vec<Hit>> {
    if matches!(token_source, TokenSource::Shingles) {
        return Err(Error::Unsupported(
            "JaccardExact: TokenSource::Shingles requires TokenSetSegment (v2)".into(),
        ));
    }
    score_set_similarity(state, frames, query_tokens, top_k, |inter, q, d| {
        let union = q + d - inter;
        if union == 0.0 { 0.0 } else { inter / union }
    })
}

pub(crate) fn dice_query(
    state: &TextSpaceState,
    frames: &[FrameStub],
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
) -> Result<Vec<Hit>> {
    score_set_similarity(state, frames, query_tokens, top_k, |inter, q, d| {
        let denom = q + d;
        if denom == 0.0 {
            0.0
        } else {
            2.0 * inter / denom
        }
    })
}

pub(crate) fn overlap_query(
    state: &TextSpaceState,
    frames: &[FrameStub],
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
) -> Result<Vec<Hit>> {
    score_set_similarity(state, frames, query_tokens, top_k, |inter, q, d| {
        let denom = q.min(d);
        if denom == 0.0 { 0.0 } else { inter / denom }
    })
}

pub(crate) fn containment_query(
    state: &TextSpaceState,
    frames: &[FrameStub],
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
) -> Result<Vec<Hit>> {
    score_set_similarity(state, frames, query_tokens, top_k, |inter, q, _d| {
        if q == 0.0 { 0.0 } else { inter / q }
    })
}

fn score_set_similarity(
    state: &TextSpaceState,
    frames: &[FrameStub],
    query_tokens: &[Vec<u8>],
    top_k: Option<usize>,
    score_fn: impl Fn(f32, f32, f32) -> f32,
) -> Result<Vec<Hit>> {
    if query_tokens.is_empty() {
        return Ok(Vec::new());
    }
    let tombstoned = build_tombstone_set(frames);

    // Distinct query term_ids that exist in the dictionary.
    let q_term_ids: Vec<TermId> = crate::retrieval::resolve_query_terms(state, query_tokens)
        .into_iter()
        .map(|(t, _)| t)
        .collect();
    if q_term_ids.is_empty() {
        return Ok(Vec::new());
    }
    let q_size = q_term_ids.len() as f32;

    // |q ∩ d| per active frame: dense u32 buffer indexed by FrameId.0,
    // touched-list for sparse collection at the end.
    let n_frames = state.doc_lengths.len();
    let mut intersection: Vec<u32> = vec![0u32; n_frames];
    let mut touched: Vec<u32> = Vec::with_capacity(256);

    for term_id in &q_term_ids {
        let postings = match state.lookup_postings_sparse(*term_id) {
            Some(p) => p,
            None => continue,
        };
        let frame_ids = postings.frame_ids.as_slice();
        for &fid in frame_ids {
            if !tombstoned.is_empty() && tombstoned.contains(&fid) {
                continue;
            }
            let idx = fid as usize;
            if idx >= intersection.len() {
                continue;
            }
            let prev = intersection[idx];
            if prev == 0 {
                touched.push(fid as u32);
            }
            intersection[idx] = prev + 1;
        }
    }

    // Materialize per-doc scores into a sparse-keyed dense f32 buffer
    // so the shared top-K helper can consume it. Only `touched` indices
    // are populated; everything else stays 0 and is filtered out by the
    // helper's `<= 0.0` guard.
    let mut scores: Vec<f32> = vec![0.0_f32; n_frames];
    for &fid32 in &touched {
        let idx = fid32 as usize;
        let inter_count = intersection[idx] as f32;
        let d_size = state.doc_unique_terms.get(idx).copied().unwrap_or(0) as f32;
        scores[idx] = score_fn(inter_count, q_size, d_size);
    }

    Ok(select_top_k_from_touched(&touched, &scores, top_k))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::file::text_indexing::PendingFrameIndex;
    use crate::format::{CollectionId, FrameId, TextSpaceId, catalog::FrameStatus};
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
    fn jaccard_known_scores() {
        // Doc 1: {alpha, beta} (|d|=2)
        // Doc 2: {alpha, beta, gamma} (|d|=3)
        // Doc 3: {delta} (|d|=1)
        // Query: {alpha, beta} (|q|=2)
        // Doc 1 inter=2 → J = 2/(2+2-2) = 1.0
        // Doc 2 inter=2 → J = 2/(2+3-2) = 0.6667
        // Doc 3 inter=0 → not present.
        let state = build_state(&[
            (1, &["alpha", "beta"]),
            (2, &["alpha", "beta", "gamma"]),
            (3, &["delta"]),
        ]);
        let frames: Vec<_> = (1..=3).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits = jaccard_query(
            &state,
            &frames,
            TokenSource::Terms,
            &tokens(&["alpha", "beta"]),
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].frame_id, FrameId(1));
        assert!((hits[0].score - 1.0).abs() < 1e-4);
        assert_eq!(hits[1].frame_id, FrameId(2));
        assert!((hits[1].score - (2.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn dice_known_scores() {
        // Doc 1: {alpha, beta} → Dice = 2*2/(2+2) = 1.0
        // Doc 2: {alpha} → Dice = 2*1/(2+1) = 0.6667
        let state = build_state(&[(1, &["alpha", "beta"]), (2, &["alpha"])]);
        let frames: Vec<_> = (1..=2).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits = dice_query(&state, &frames, &tokens(&["alpha", "beta"]), None).unwrap();
        assert!((hits[0].score - 1.0).abs() < 1e-4);
        assert!((hits[1].score - (2.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn overlap_uses_min_size() {
        // Query: {alpha, beta} (|q|=2)
        // Doc 1: {alpha, beta, gamma, delta} (|d|=4) → inter=2, min=2, overlap = 2/2 = 1.0
        // Doc 2: {alpha} (|d|=1) → inter=1, min=1, overlap = 1/1 = 1.0
        let state = build_state(&[(1, &["alpha", "beta", "gamma", "delta"]), (2, &["alpha"])]);
        let frames: Vec<_> = (1..=2).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits = overlap_query(&state, &frames, &tokens(&["alpha", "beta"]), None).unwrap();
        for h in &hits {
            assert!((h.score - 1.0).abs() < 1e-4);
        }
    }

    #[test]
    fn containment_normalizes_by_query_size() {
        // Query: {alpha, beta, gamma} (|q|=3)
        // Doc 1: {alpha, beta} → inter=2, containment = 2/3
        // Doc 2: {alpha, beta, gamma, delta} → inter=3, containment = 3/3 = 1.0
        let state = build_state(&[
            (1, &["alpha", "beta"]),
            (2, &["alpha", "beta", "gamma", "delta"]),
        ]);
        let frames: Vec<_> = (1..=2).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits =
            containment_query(&state, &frames, &tokens(&["alpha", "beta", "gamma"]), None).unwrap();
        // Frame 2 fully contains the query → score 1.0; frame 1 partial → 2/3.
        assert_eq!(hits[0].frame_id, FrameId(2));
        assert!((hits[0].score - 1.0).abs() < 1e-4);
        assert_eq!(hits[1].frame_id, FrameId(1));
        assert!((hits[1].score - (2.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn rejects_shingles_token_source_as_v2() {
        let state = build_state(&[(1, &["alpha"])]);
        let frames = vec![frame(1, FrameStatus::Active)];
        let err = jaccard_query(
            &state,
            &frames,
            TokenSource::Shingles,
            &tokens(&["alpha"]),
            None,
        )
        .expect_err("shingles is v2");
        assert!(err.to_string().contains("Shingles"));
    }

    #[test]
    fn tombstoned_frames_excluded() {
        let state = build_state(&[(1, &["alpha"]), (2, &["alpha"])]);
        let frames = vec![
            frame(1, FrameStatus::Active),
            frame(2, FrameStatus::Tombstoned),
        ];
        let hits = jaccard_query(
            &state,
            &frames,
            TokenSource::Terms,
            &tokens(&["alpha"]),
            None,
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].frame_id, FrameId(1));
    }

    #[test]
    fn top_k_caps_jaccard_results() {
        let state = build_state(&[
            (1, &["alpha", "beta"]),
            (2, &["alpha", "beta", "gamma"]),
            (3, &["alpha"]),
        ]);
        let frames: Vec<_> = (1..=3).map(|i| frame(i, FrameStatus::Active)).collect();
        let hits = jaccard_query(
            &state,
            &frames,
            TokenSource::Terms,
            &tokens(&["alpha", "beta"]),
            Some(1),
        )
        .unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].frame_id, FrameId(1));
    }

    #[test]
    fn empty_query_yields_no_hits() {
        let state = build_state(&[(1, &["alpha"])]);
        let frames = vec![frame(1, FrameStatus::Active)];
        for hits in [
            jaccard_query(&state, &frames, TokenSource::Terms, &[], None).unwrap(),
            dice_query(&state, &frames, &[], None).unwrap(),
            overlap_query(&state, &frames, &[], None).unwrap(),
            containment_query(&state, &frames, &[], None).unwrap(),
        ] {
            assert!(hits.is_empty());
        }
    }
}
