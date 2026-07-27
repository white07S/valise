//! Score normalization and fusion across retrieval channels, spec §18.3.
//!
//! Channels in v1: BM25/TF-IDF/Jaccard text scores (higher = better) and
//! ANN vector distance (smaller = better). Fusion converts everything to
//! a single "higher = better" axis before combining: vector distances
//! become `-distance` so weighted sums move in the same direction.

use std::collections::HashMap;

use crate::format::{
    FrameId,
    catalog::{FusionNormalization, FusionProfileDesc},
};

/// Apply z-score normalization to a column of values, ignoring None.
fn zscore(values: Vec<Option<f32>>) -> Vec<Option<f32>> {
    let present: Vec<f32> = values.iter().filter_map(|v| *v).collect();
    if present.is_empty() {
        return values;
    }
    let mean: f32 = present.iter().sum::<f32>() / present.len() as f32;
    let var: f32 = present.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / present.len() as f32;
    let std = var.sqrt().max(f32::EPSILON);
    values
        .into_iter()
        .map(|v| v.map(|x| (x - mean) / std))
        .collect()
}

fn minmax(values: Vec<Option<f32>>) -> Vec<Option<f32>> {
    let present: Vec<f32> = values.iter().filter_map(|v| *v).collect();
    if present.is_empty() {
        return values;
    }
    let min = present.iter().copied().fold(f32::INFINITY, f32::min);
    let max = present.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denom = (max - min).max(f32::EPSILON);
    values
        .into_iter()
        .map(|v| v.map(|x| (x - min) / denom))
        .collect()
}

/// RRF column: convert ranks (lower = better) to scores via 1/(k + rank).
/// Rank is determined by sorting present values ascending or descending —
/// since our channel convention is higher = better, we sort descending.
fn rrf_column(values: Vec<Option<f32>>, k_smoothing: u32) -> Vec<Option<f32>> {
    let n = values.len();
    let mut indexed: Vec<(usize, f32)> = values
        .iter()
        .enumerate()
        .filter_map(|(i, v)| v.map(|x| (i, x)))
        .collect();
    // Higher score = better → sort descending so rank 0 is the top hit.
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
    let mut rrf = vec![None; n];
    for (rank, (i, _)) in indexed.iter().enumerate() {
        // RRF score: 1 / (k + rank + 1), using 1-based rank to match
        // standard RRF (k=60 ⇒ top hit scores 1/61 ≈ 0.0164).
        let r = (rank + 1) as f32;
        rrf[*i] = Some(1.0 / (k_smoothing as f32 + r));
    }
    rrf
}

/// Normalize one channel column according to `mode`. `Rrf` is rank-based
/// so it ignores absolute values; the others rescale magnitudes.
fn normalize_column(values: Vec<Option<f32>>, mode: FusionNormalization) -> Vec<Option<f32>> {
    match mode {
        FusionNormalization::None => values,
        FusionNormalization::ZScore => zscore(values),
        FusionNormalization::MinMax => minmax(values),
        FusionNormalization::Rrf { k } => rrf_column(values, k),
    }
}

/// Compute a fused score per frame_id given each channel's raw per-frame
/// scores. The output is sorted descending by fused score.
///
/// `vector_distances` enter as raw codec distances (smaller = better);
/// the function flips them to `-distance` before normalization so all
/// channels share the "higher = better" direction.
pub(crate) fn fuse(
    profile: &FusionProfileDesc,
    bm25: HashMap<FrameId, f32>,
    tfidf: HashMap<FrameId, f32>,
    jaccard: HashMap<FrameId, f32>,
    vector_distances: HashMap<FrameId, f32>,
) -> Vec<(FrameId, f32)> {
    // Union of all candidate frame_ids in deterministic order (ascending).
    let mut frame_ids: Vec<FrameId> = bm25
        .keys()
        .chain(tfidf.keys())
        .chain(jaccard.keys())
        .chain(vector_distances.keys())
        .copied()
        .collect();
    frame_ids.sort_by_key(|f| f.0);
    frame_ids.dedup();
    if frame_ids.is_empty() {
        return Vec::new();
    }

    let bm25_col: Vec<Option<f32>> = frame_ids.iter().map(|f| bm25.get(f).copied()).collect();
    let tfidf_col: Vec<Option<f32>> = frame_ids.iter().map(|f| tfidf.get(f).copied()).collect();
    let jaccard_col: Vec<Option<f32>> = frame_ids.iter().map(|f| jaccard.get(f).copied()).collect();
    let vector_col: Vec<Option<f32>> = frame_ids
        .iter()
        .map(|f| vector_distances.get(f).map(|d| -d))
        .collect();

    let bm25_n = normalize_column(bm25_col, profile.normalization);
    let tfidf_n = normalize_column(tfidf_col, profile.normalization);
    let jaccard_n = normalize_column(jaccard_col, profile.normalization);
    let vector_n = normalize_column(vector_col, profile.normalization);

    let mut results: Vec<(FrameId, f32)> = Vec::with_capacity(frame_ids.len());
    for (i, fid) in frame_ids.iter().enumerate() {
        let mut score = 0.0f32;
        let mut weight_sum = 0.0f32;
        for (val_opt, weight) in [
            (bm25_n[i], profile.bm25_weight),
            (tfidf_n[i], profile.tfidf_weight),
            (jaccard_n[i], profile.jaccard_weight),
            (vector_n[i], profile.vector_weight),
        ] {
            if weight == 0.0 {
                continue;
            }
            if let Some(v) = val_opt {
                score += weight * v;
                weight_sum += weight;
            }
        }
        // If no channel produced a score for this frame, drop it.
        if weight_sum == 0.0 {
            continue;
        }
        results.push((*fid, score));
    }
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::FusionProfileId;

    fn profile(norm: FusionNormalization, w: (f32, f32, f32, f32)) -> FusionProfileDesc {
        FusionProfileDesc {
            fusion_profile_id: FusionProfileId(1),
            bm25_weight: w.0,
            tfidf_weight: w.1,
            jaccard_weight: w.2,
            vector_weight: w.3,
            normalization: norm,
        }
    }

    fn frame_score_map(items: &[(u64, f32)]) -> HashMap<FrameId, f32> {
        items.iter().map(|(f, s)| (FrameId(*f), *s)).collect()
    }

    #[test]
    fn fuse_with_none_normalization_uses_raw_scores() {
        let prof = profile(FusionNormalization::None, (1.0, 0.0, 0.0, 1.0));
        let bm25 = frame_score_map(&[(1, 5.0), (2, 1.0)]);
        let vec = frame_score_map(&[(1, 0.1), (2, 0.5)]);
        let out = fuse(&prof, bm25, HashMap::new(), HashMap::new(), vec);
        // Frame 1: 5.0 - 0.1 = 4.9; Frame 2: 1.0 - 0.5 = 0.5.
        // Ordered descending → (1, 4.9), (2, 0.5).
        assert_eq!(out[0].0, FrameId(1));
        assert!((out[0].1 - 4.9).abs() < 1e-5);
        assert_eq!(out[1].0, FrameId(2));
    }

    #[test]
    fn fuse_with_zscore_neutralizes_magnitude_disparity() {
        // BM25 scale ≫ vector scale; without normalization BM25 dominates.
        // After ZScore the contribution of each is ~unit-scale.
        let prof = profile(FusionNormalization::ZScore, (1.0, 0.0, 0.0, 1.0));
        let bm25 = frame_score_map(&[(1, 100.0), (2, 50.0), (3, 1.0)]);
        let vec = frame_score_map(&[(1, 0.5), (2, 0.1), (3, 0.05)]);
        let out = fuse(&prof, bm25, HashMap::new(), HashMap::new(), vec);
        // Frame 1 has high BM25 but worst vector distance.
        // Frame 2 has medium BM25 and best non-trivial vector.
        // Expectation isn't a strict rank; we just assert no NaN and length.
        assert_eq!(out.len(), 3);
        for (_, s) in &out {
            assert!(s.is_finite());
        }
    }

    #[test]
    fn fuse_with_minmax_keeps_scores_in_unit_interval_pre_weight() {
        let prof = profile(FusionNormalization::MinMax, (1.0, 0.0, 0.0, 0.0));
        let bm25 = frame_score_map(&[(1, 10.0), (2, 5.0), (3, 0.0)]);
        let out = fuse(&prof, bm25, HashMap::new(), HashMap::new(), HashMap::new());
        // After MinMax: 1.0, 0.5, 0.0. Weighted by 1.0 → same.
        // Order: 1, 2, 3.
        assert_eq!(out[0].0, FrameId(1));
        assert!((out[0].1 - 1.0).abs() < 1e-5);
        assert_eq!(out[1].0, FrameId(2));
        assert!((out[1].1 - 0.5).abs() < 1e-5);
        assert_eq!(out[2].0, FrameId(3));
        assert!(out[2].1.abs() < 1e-5);
    }

    #[test]
    fn fuse_with_rrf_uses_ranks_not_magnitudes() {
        let prof = profile(FusionNormalization::Rrf { k: 60 }, (1.0, 0.0, 0.0, 1.0));
        let bm25 = frame_score_map(&[(1, 100.0), (2, 50.0)]);
        let vec = frame_score_map(&[(1, 0.9), (2, 0.1)]); // Frame 2 wins on vector
        let out = fuse(&prof, bm25, HashMap::new(), HashMap::new(), vec);
        // Frame 1: bm25 rank 1, vector rank 2 → 1/61 + 1/62
        // Frame 2: bm25 rank 2, vector rank 1 → 1/62 + 1/61
        // Both should fuse to the same score.
        assert_eq!(out.len(), 2);
        assert!((out[0].1 - out[1].1).abs() < 1e-6);
    }

    #[test]
    fn fuse_drops_frames_with_no_scoring_channel() {
        // Only vector_weight is non-zero, but Frame 2 only has BM25 scores.
        let prof = profile(FusionNormalization::None, (0.0, 0.0, 0.0, 1.0));
        let bm25 = frame_score_map(&[(2, 99.0)]);
        let vec = frame_score_map(&[(1, 0.5)]);
        let out = fuse(&prof, bm25, HashMap::new(), HashMap::new(), vec);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].0, FrameId(1));
    }
}
