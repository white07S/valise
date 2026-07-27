//! Query-input and query-result value types for the `ValiseFile` engine
//! surface (text / vector / hybrid / time). Re-exported from `file.rs` so
//! the public path stays `crate::file::<Type>`.

use std::collections::HashSet;

use super::api_types::VectorFidelity;
use crate::format::catalog::TfMode;
use crate::format::{
    CollectionId, EmbeddingSpaceId, FrameId, FusionProfileId, RetrievalProfileId, TextSpaceId,
    VectorId,
};

/// Text query input passed to `ValiseFile::query_text`.
#[derive(Clone, Debug)]
pub struct TextQuery {
    pub text_space_id: TextSpaceId,
    pub query: String,
    pub algorithm: QueryAlgorithm,
    /// Optional cap on returned hits. When `Some(k)`, the retriever
    /// uses a bounded-K heap inside the algorithm — much cheaper than
    /// sorting the entire touched set and truncating at the boundary.
    /// `None` keeps the legacy behavior of returning every hit with a
    /// non-zero score, sorted.
    pub top_k: Option<usize>,
    /// Per-term posting-list read budget for the impact-sorted
    /// vote-then-rerank pipeline used by the cosine scorers. Posting
    /// lists are sorted by descending term-frequency at decode time;
    /// the vote phase reads at most `channel_k` entries per query
    /// term, which bounds the candidate set to `|q| × channel_k`
    /// regardless of corpus size. The rerank phase then recomputes
    /// the exact score (full idf / normalization / L2) only for the
    /// top rerank candidates via binary search on the original
    /// frame_id-sorted posting list. `None` disables the cap and
    /// retains exact full-posting scoring.
    pub channel_k: Option<usize>,
}

/// Algorithm dispatch for `query_text`. Profile-driven variants pull their
/// scoring parameters (k1/b, tf_mode/idf_mode/norm_mode, token_source) from
/// the registered `RetrievalProfileDesc`. Profile-free variants are derived
/// directly from the canonical primitives per spec §12.0.5 — no profile
/// registration required.
#[derive(Clone, Copy, Debug)]
pub enum QueryAlgorithm {
    Profile(RetrievalProfileId),
    /// Profile-free BM25. `k1`/`b` are supplied per query and scoring is
    /// derived from the canonical df / doc-length / avgdl in
    /// `TextSpaceState` — no retrieval profile, hence no catalog write on
    /// the read path (unlike `Profile(Bm25)`). Uses the standard
    /// Robertson/Sparck-Jones idf. Defaults: `k1 = 1.2`, `b = 0.75`.
    Bm25 {
        k1: f32,
        b: f32,
    },
    CountCosine,
    TfidfCosine {
        tf_mode: TfMode,
    },
    /// Count cosine using `sqrt(doc_length)` as the per-doc norm
    /// instead of the exact `sqrt(sum_t tf(t,d)^2)`. Lucene-style
    /// length proxy — trades a small NDCG drift for eliminating the
    /// index-wide L2 sweep.
    CountCosineApprox,
    /// tf-idf cosine with `sqrt(doc_length)` per-doc norm. Same
    /// trade-off as [`Self::CountCosineApprox`].
    TfidfCosineApprox {
        tf_mode: TfMode,
    },
    Dice,
    Overlap,
    Containment,
}

/// Default candidate budget for the sign-sketch scan when `channel_k` is
/// `None`. Fixed and corpus-size-INDEPENDENT on purpose: the sketch's
/// coverage of the true top-k saturates by a couple thousand candidates at
/// d≈768 (sweeps 06/14; see `docs/VECTOR_SEARCH.md`), so scaling the budget
/// with N only multiplies rerank cost for no recall gain. Callers that need a
/// different operating point (very large N, or higher recall) pass an explicit
/// `channel_k`.
pub(crate) const DEFAULT_SKETCH_CANDIDATE_BUDGET: usize = 2048;

/// Input to [`ValiseFile::vector_search`].
///
/// Dispatch is implicit per embedding space: a QAM(5,6) space uses the
/// in-memory sign-sketch scan + QAM-sliding rerank; any other space falls back
/// to a full brute-force scan through the primary codec. See
/// `docs/VECTOR_SEARCH.md` for the pipeline and its recall/latency envelope.
#[derive(Clone, Debug)]
pub struct VectorSearchQuery {
    pub embedding_space_id: EmbeddingSpaceId,
    pub query: Vec<f32>,
    pub k: usize,
    /// Candidate budget for the sign-sketch scan: it keeps the `channel_k`
    /// closest-by-Hamming vectors as rerank candidates. `None` uses
    /// `max(4 * k, DEFAULT_SKETCH_CANDIDATE_BUDGET)`, clamped to the active
    /// count. Brute force (non-QAM(5,6) spaces) ignores it.
    pub channel_k: Option<usize>,
    /// Optional collection allowlist.
    pub collection_filter: Option<HashSet<CollectionId>>,
    /// Rerank fidelity over the sketch candidates. `Lossy` keeps the
    /// QAM-sliding (i8) scores; `Full` adds an f32 rerank pass over the top
    /// survivors. Recall is bounded by the codec's reconstruction either way
    /// (see `docs/VECTOR_SEARCH.md`); `Full` recovers the last ~3 points the
    /// i8 scorer leaves on the table. Neither changes the candidate set — that
    /// is `channel_k`'s job.
    pub fidelity: VectorFidelity,
}

impl VectorSearchQuery {
    /// Accurate search: sketch-select the default candidate budget, then run
    /// the f32 rerank pass (`VectorFidelity::Full`). Leaves `channel_k` unset
    /// so it uses [`DEFAULT_SKETCH_CANDIDATE_BUDGET`], which is sized to reach
    /// the codec's recall ceiling. Pass an explicit `channel_k` to trade
    /// recall for latency. This is the engine-level target a higher layer's
    /// "accurate rerank" lowers to.
    #[must_use]
    pub fn accurate(embedding_space_id: EmbeddingSpaceId, query: Vec<f32>, k: usize) -> Self {
        Self {
            embedding_space_id,
            query,
            k,
            channel_k: None,
            collection_filter: None,
            fidelity: VectorFidelity::Full,
        }
    }

    /// Fast search: sketch-select the default candidate budget, keep the
    /// QAM-sliding (i8) scores, no f32 rerank (`VectorFidelity::Lossy`).
    /// ~3 points lower recall than [`Self::accurate`] for less rerank work.
    #[must_use]
    pub fn fast(embedding_space_id: EmbeddingSpaceId, query: Vec<f32>, k: usize) -> Self {
        Self {
            embedding_space_id,
            query,
            k,
            channel_k: None,
            collection_filter: None,
            fidelity: VectorFidelity::Lossy,
        }
    }
}

/// Vector hit produced by [`ValiseFile::vector_search`]. Score follows
/// the "smaller = more similar" codec convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VectorHit {
    pub vector_id: VectorId,
    pub frame_id: FrameId,
    pub collection_id: CollectionId,
    pub score: f32,
}

/// Per-stage wall-clock breakdown of a [`ValiseFile::vector_search_traced`]
/// call. Sum of fields ≈ total call cost. `rerank_full` is zero when the
/// caller did not request [`VectorFidelity::Full`].
#[derive(Clone, Copy, Debug, Default)]
pub struct VoteSearchTrace {
    /// Codec + space resolution.
    pub preflight: std::time::Duration,
    /// Unused (sketch query encoding is folded into the scan stage).
    pub encode: std::time::Duration,
    /// Unused (the sketch index is derived at file-open, not per query).
    pub resolve: std::time::Duration,
    /// Candidate generation: the sign-sketch Hamming scan + counting-sort.
    pub vote: SketchScanTimings,
    /// QAM-sliding rerank pass over the candidates.
    pub rerank_int: std::time::Duration,
    /// Optional f32 re-scoring pass. Zero unless `query.fidelity ==
    /// VectorFidelity::Full`.
    pub rerank_full: std::time::Duration,
}

/// Sub-stage timings for the sign-sketch candidate scan.
#[derive(Clone, Copy, Debug, Default)]
pub struct SketchScanTimings {
    /// Fused Hamming scan + histogram + counting-sort threshold.
    pub accumulate: std::time::Duration,
}

/// Input to [`ValiseFile::time_range_query`], spec §17. `from` and `to` are
/// inclusive Unix epoch seconds. Empty range (`from > to`) yields no hits.
#[derive(Clone, Copy, Debug)]
pub struct TimeQuery {
    pub from: i64,
    pub to: i64,
    /// Restrict to one collection. `None` returns hits from all collections.
    pub collection_id: Option<CollectionId>,
}

/// Input to [`ValiseFile::query_hybrid`], spec §18.3.
///
/// One text channel (running an indexed retrieval profile) and one vector
/// channel are fused under `fusion_profile_id`. Either channel may be left
/// unset, in which case the fused score is just the other channel.
#[derive(Clone, Debug)]
pub struct HybridQuery {
    pub fusion_profile_id: FusionProfileId,
    /// Text channel: omit to skip text retrieval entirely.
    pub text: Option<HybridTextChannel>,
    /// Vector channel: omit to skip vector retrieval entirely.
    pub vector: Option<HybridVectorChannel>,
    pub k: usize,
    /// Per-channel candidate budget. Each channel pulls this many hits
    /// before fusion picks the top-k.
    pub channel_k: usize,
}

#[derive(Clone, Debug)]
pub struct HybridTextChannel {
    pub text_space_id: TextSpaceId,
    pub query: String,
    pub algorithm: QueryAlgorithm,
}

#[derive(Clone, Debug)]
pub struct HybridVectorChannel {
    pub embedding_space_id: EmbeddingSpaceId,
    pub query: Vec<f32>,
    pub ef: Option<usize>,
    pub fidelity: VectorFidelity,
}

/// Fused hit produced by [`ValiseFile::query_hybrid`]. `score` follows the
/// "higher = better" convention (vector distances are flipped before
/// fusion so all channels share direction).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HybridHit {
    pub frame_id: FrameId,
    pub score: f32,
}
