//! Query-time retrieval modules. Text retrieval (BM25, TF-IDF, exact
//! term-Jaccard) over the canonical primitives produced by the indexing
//! pipeline, plus the QAM sign-sketch scan + QAM-sliding rerank for
//! vectors (see [`sketch`]).

pub mod bm25;
pub mod fusion;
pub mod jaccard;
pub mod sketch;
pub mod tfidf;
pub mod top_k;

/// Bench-only entry points for the sketch popcount kernels. Gated
/// behind the `bench` cargo feature.
///
/// - `hamming` — the per-vector dispatcher (aarch64 → NEON; on
///   x86_64 today this is the scalar reference, since §4.2 routes
///   AVX2 through `scan_candidates_avx2`, NOT through the per-vector
///   path).
/// - `hamming_scalar` — always-available scalar reference.
/// - `hamming_avx2_popcnt`, `hamming_avx2_shuffle` (x86_64 only) —
///   the two AVX2 variants that ship inside `scan_candidates_avx2`.
///   Exposed here so the criterion bench can time them in isolation
///   against `hamming_scalar`; callers MUST verify
///   `is_x86_feature_detected!("avx2")` first.
#[cfg(any(test, feature = "bench"))]
pub mod sketch_bench {
    use crate::retrieval::sketch;

    #[inline(always)]
    pub fn hamming(a: &[u64], b: &[u64]) -> u32 {
        sketch::hamming(a, b)
    }

    #[inline(always)]
    pub fn hamming_scalar(a: &[u64], b: &[u64]) -> u32 {
        sketch::hamming_scalar(a, b)
    }

    /// # Safety
    ///
    /// Caller must have verified `avx2` and `popcnt` CPU support before
    /// calling.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub unsafe fn hamming_avx2_popcnt(a: &[u64], b: &[u64]) -> u32 {
        unsafe { sketch::hamming_x86_64_bench::hamming_popcnt(a, b) }
    }

    /// # Safety
    ///
    /// Caller must have verified `avx2` and `popcnt` CPU support before
    /// calling.
    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    pub unsafe fn hamming_avx2_shuffle(a: &[u64], b: &[u64]) -> u32 {
        unsafe { sketch::hamming_x86_64_bench::hamming_shuffle(a, b) }
    }
}

use crate::file::text_indexing::TextSpaceState;
use crate::format::{FrameId, TermId};

/// Search hit produced by a retrieval algorithm. Score convention is
/// algorithm-specific: BM25 / TF-IDF / cosine / Jaccard / Dice / overlap /
/// containment all use higher = better. Each algorithm's doc comment
/// pins the convention.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Hit {
    pub frame_id: FrameId,
    pub score: f32,
}

/// Resolve analyzer tokens to `(TermId, query_frequency)` pairs in
/// first-seen order, dropping tokens absent from the term dictionary.
///
/// The shared front half of every lexical scorer — resolve once, then
/// fetch per-term stats (`df_for` / `cf_for` / postings) and score. New
/// algorithms should start here; see `docs/EXTENDING.md`.
pub(crate) fn resolve_query_terms(
    state: &TextSpaceState,
    query_tokens: &[Vec<u8>],
) -> Vec<(TermId, u32)> {
    let mut qfs: Vec<(TermId, u32)> = Vec::with_capacity(query_tokens.len());
    for token in query_tokens {
        if let Some(term_id) = state.lookup_term(token) {
            match qfs.iter_mut().find(|(t, _)| *t == term_id) {
                Some((_, qf)) => *qf += 1,
                None => qfs.push((term_id, 1)),
            }
        }
    }
    qfs
}
