//! The read-side query model: the [`Search`] builder and its value types
//! (`TextScorer`, `Fusion`, `Rerank`, `Recency`, `Hit`). Execution lives in
//! `reader.rs`.
//!
//! Algorithm choice is per-query (DB-layer plan §9.A-4) and the entire read
//! path is **write-free**: every `TextScorer` lowers to a profile-free
//! `QueryAlgorithm`, and fusion is computed layer-side, so a search never
//! registers a catalog object. (Exact Jaccard, the one scorer that would need
//! a registered profile, is intentionally absent in v1 — use `Dice`.)

use crate::db::record::Key;
use crate::file::QueryAlgorithm;
pub use crate::format::catalog::TfMode;

/// Per-query text scoring algorithm. Every variant is profile-free (lowers to a
/// `QueryAlgorithm` that reads canonical stats without a catalog write).
///
/// Prefer the constructor fns ([`TextScorer::bm25`], …) over variant
/// literals; the enum is `#[non_exhaustive]` so scorers can be added without
/// a breaking change.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum TextScorer {
    /// Okapi BM25. `Self::bm25()` gives the standard `k1=1.2, b=0.75`.
    Bm25 {
        k1: f32,
        b: f32,
    },
    TfidfCosine {
        tf_mode: TfMode,
    },
    CountCosine,
    /// `sqrt(doclen)`-normalized BM25-ish approximation (cheaper).
    TfidfCosineApprox {
        tf_mode: TfMode,
    },
    CountCosineApprox,
    /// Sørensen–Dice set similarity. The profile-free stand-in for exact
    /// Jaccard (which requires a registered profile and is deferred in v1).
    Dice,
    /// `|q ∩ d| / min(|q|, |d|)` set overlap.
    Overlap,
    /// `|q ∩ d| / |q|` — fraction of the query's terms found in the doc.
    Containment,
}

impl TextScorer {
    /// Okapi BM25 with the standard parameters (`k1=1.2, b=0.75`) — the
    /// default scorer of [`Search::text`].
    #[must_use]
    pub fn bm25() -> Self {
        TextScorer::Bm25 { k1: 1.2, b: 0.75 }
    }

    /// Okapi BM25 with explicit parameters.
    #[must_use]
    pub fn bm25_with(k1: f32, b: f32) -> Self {
        TextScorer::Bm25 { k1, b }
    }

    /// TF-IDF cosine over the canonical statistics.
    #[must_use]
    pub fn tfidf_cosine(tf_mode: TfMode) -> Self {
        TextScorer::TfidfCosine { tf_mode }
    }

    /// Cheaper `sqrt(doclen)`-normalized TF-IDF cosine approximation.
    #[must_use]
    pub fn tfidf_cosine_approx(tf_mode: TfMode) -> Self {
        TextScorer::TfidfCosineApprox { tf_mode }
    }

    /// Raw term-count cosine.
    #[must_use]
    pub fn count_cosine() -> Self {
        TextScorer::CountCosine
    }

    /// Cheaper count-cosine approximation.
    #[must_use]
    pub fn count_cosine_approx() -> Self {
        TextScorer::CountCosineApprox
    }

    /// Sørensen–Dice set similarity.
    #[must_use]
    pub fn dice() -> Self {
        TextScorer::Dice
    }

    /// `|q ∩ d| / min(|q|, |d|)` set overlap.
    #[must_use]
    pub fn overlap() -> Self {
        TextScorer::Overlap
    }

    /// `|q ∩ d| / |q|` query-term containment.
    #[must_use]
    pub fn containment() -> Self {
        TextScorer::Containment
    }

    pub(crate) fn to_algorithm(self) -> QueryAlgorithm {
        match self {
            TextScorer::Bm25 { k1, b } => QueryAlgorithm::Bm25 { k1, b },
            TextScorer::TfidfCosine { tf_mode } => QueryAlgorithm::TfidfCosine { tf_mode },
            TextScorer::CountCosine => QueryAlgorithm::CountCosine,
            TextScorer::TfidfCosineApprox { tf_mode } => {
                QueryAlgorithm::TfidfCosineApprox { tf_mode }
            }
            TextScorer::CountCosineApprox => QueryAlgorithm::CountCosineApprox,
            TextScorer::Dice => QueryAlgorithm::Dice,
            TextScorer::Overlap => QueryAlgorithm::Overlap,
            TextScorer::Containment => QueryAlgorithm::Containment,
        }
    }
}

/// How to combine ≥2 channels. With a single channel, fusion is a no-op (the
/// channel's own ranking is used). Construct via [`Fusion::rrf`] /
/// [`Fusion::weighted`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Fusion {
    /// Reciprocal Rank Fusion. Per channel a frame at 0-based rank `p`
    /// contributes `1/(k + p + 1)`; contributions are **summed** across
    /// channels (a frame missing from a channel contributes 0). `k≈60` is the
    /// usual smoothing constant.
    Rrf { k: u32 },
    /// Min-max normalize each channel to `[0,1]`, then weighted sum. The
    /// `vector` weight applies to every vector channel.
    Weighted { text: f32, vector: f32 },
}

impl Fusion {
    /// Reciprocal Rank Fusion with smoothing constant `k` (`k=60` is the
    /// default of [`Search::new`]).
    #[must_use]
    pub fn rrf(k: u32) -> Self {
        Fusion::Rrf { k }
    }

    /// Min-max-normalized weighted sum with per-channel-kind weights.
    #[must_use]
    pub fn weighted(text: f32, vector: f32) -> Self {
        Fusion::Weighted { text, vector }
    }
}

impl Default for Fusion {
    fn default() -> Self {
        Fusion::Rrf { k: 60 }
    }
}

/// Vector rerank fidelity. `Accurate` (the default of [`Search::vector`])
/// forces the vote-then-rerank path with a full f32 rerank (defusing the
/// `VectorFidelity::Full` footgun, plan §9.C); `Fast` keeps quantized scores.
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Rerank {
    Fast,
    Accurate,
}

/// Time-awareness. See DB-layer plan §6 "Temporal model". Construct via
/// [`Recency::range`] / [`Recency::half_life`] / [`Recency::rrf_channel`].
#[derive(Clone, Copy, Debug)]
#[non_exhaustive]
pub enum Recency {
    /// Hard filter: prune candidates outside `[from, to]` (inclusive unix
    /// seconds) before fusion. NB: the engine has no in-query time filter, so a
    /// very narrow window over a large corpus can return fewer than `top_k`.
    Range { from: i64, to: i64 },
    /// Soft post-fusion multiplier: `score · 0.5^(age / half_life)`. Composes
    /// with any fusion.
    HalfLife { days: f64 },
    /// Soft: recency as an additional RRF channel (candidates ranked newest
    /// first, contributing `weight/(k + rank + 1)`). Requires `Fusion::Rrf`.
    RrfChannel { half_life_days: f64, weight: f32 },
}

impl Recency {
    /// Hard filter to `[from, to]` (inclusive unix seconds).
    #[must_use]
    pub fn range(from: i64, to: i64) -> Self {
        Recency::Range { from, to }
    }

    /// Soft post-fusion half-life decay multiplier.
    #[must_use]
    pub fn half_life(days: f64) -> Self {
        Recency::HalfLife { days }
    }

    /// Soft recency RRF channel (requires [`Fusion::rrf`]).
    #[must_use]
    pub fn rrf_channel(half_life_days: f64, weight: f32) -> Self {
        Recency::RrfChannel {
            half_life_days,
            weight,
        }
    }
}

/// One search hit. Field values are not inlined; fetch them with
/// [`crate::db::Reader::get`] / `get_many` (the key is always resolvable).
#[derive(Clone, Debug)]
pub struct Hit {
    pub key: Key,
    pub score: f32,
    pub collection: String,
}

/// The hits of one search, best-first. Derefs to `[Hit]` and iterates (owned
/// or by reference), so it can be used wherever a slice/`Vec` of hits was.
///
/// `#[non_exhaustive]` keeps construction crate-internal so result-level
/// metadata (timings, counters, …) can be added later without a breaking
/// change.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct SearchResult {
    pub hits: Vec<Hit>,
}

impl SearchResult {
    pub(crate) fn new(hits: Vec<Hit>) -> Self {
        Self { hits }
    }

    /// Consume the result, returning the owned hits.
    #[must_use]
    pub fn into_hits(self) -> Vec<Hit> {
        self.hits
    }
}

impl std::ops::Deref for SearchResult {
    type Target = [Hit];
    fn deref(&self) -> &[Hit] {
        &self.hits
    }
}

impl IntoIterator for SearchResult {
    type Item = Hit;
    type IntoIter = std::vec::IntoIter<Hit>;
    fn into_iter(self) -> Self::IntoIter {
        self.hits.into_iter()
    }
}

impl<'a> IntoIterator for &'a SearchResult {
    type Item = &'a Hit;
    type IntoIter = std::slice::Iter<'a, Hit>;
    fn into_iter(self) -> Self::IntoIter {
        self.hits.iter()
    }
}

pub(crate) struct TextChannelSpec {
    pub field: String,
    pub query: String,
    pub scorer: TextScorer,
}

pub(crate) struct VectorChannelSpec {
    pub field: String,
    pub query: Vec<f32>,
    pub rerank: Rerank,
}

/// Fluent search specification. Target text/vector fields by name, pick a
/// fusion, optionally add recency, and bound `top_k`.
#[derive(Default)]
pub struct Search {
    pub(crate) text: Option<TextChannelSpec>,
    pub(crate) vectors: Vec<VectorChannelSpec>,
    pub(crate) fusion: Fusion,
    pub(crate) recency: Option<Recency>,
    pub(crate) top_k: usize,
    pub(crate) now_override: Option<i64>,
}

impl Search {
    #[must_use]
    pub fn new() -> Self {
        Self {
            text: None,
            vectors: Vec::new(),
            fusion: Fusion::default(),
            recency: None,
            top_k: 10,
            now_override: None,
        }
    }

    /// Search a text field with the default scorer ([`TextScorer::bm25`]).
    /// At most one text field per collection in v1, so calling this twice
    /// replaces the prior text channel.
    #[must_use]
    pub fn text(self, field: &str, query: &str) -> Self {
        self.text_with(field, query, TextScorer::bm25())
    }

    /// [`Search::text`] with an explicit scorer.
    #[must_use]
    pub fn text_with(mut self, field: &str, query: &str, scorer: TextScorer) -> Self {
        self.text = Some(TextChannelSpec {
            field: field.to_owned(),
            query: query.to_owned(),
            scorer,
        });
        self
    }

    /// Add a vector channel with the default fidelity ([`Rerank::Accurate`]).
    /// May be called multiple times for multi-vector / multimodal search;
    /// each becomes a fusion channel.
    #[must_use]
    pub fn vector(self, field: &str, query: &[f32]) -> Self {
        self.vector_with(field, query, Rerank::Accurate)
    }

    /// [`Search::vector`] with an explicit rerank fidelity.
    #[must_use]
    pub fn vector_with(mut self, field: &str, query: &[f32], rerank: Rerank) -> Self {
        self.vectors.push(VectorChannelSpec {
            field: field.to_owned(),
            query: query.to_vec(),
            rerank,
        });
        self
    }

    /// Set the fusion strategy (only meaningful with ≥2 channels).
    #[must_use]
    pub fn fuse(mut self, fusion: Fusion) -> Self {
        self.fusion = fusion;
        self
    }

    #[must_use]
    pub fn recency(mut self, recency: Recency) -> Self {
        self.recency = Some(recency);
        self
    }

    #[must_use]
    pub fn top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    /// Override "now" (unix seconds) for `Recency::HalfLife`/`RrfChannel` decay.
    /// Defaults to the system clock; set this for deterministic results.
    #[must_use]
    pub fn now(mut self, unix_secs: i64) -> Self {
        self.now_override = Some(unix_secs);
        self
    }
}
