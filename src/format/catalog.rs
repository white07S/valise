//! Core catalog descriptors, spec §10.

use crate::{
    error::{Error, Result},
    format::{
        AnalyzerId, Checksum, CodecId, CollectionId, EmbeddingSpaceId, FieldSchemaId, FrameId,
        FusionProfileId, RetrievalProfileId, SegmentId, TextSpaceId, VectorId, payload::PayloadRef,
        segment::SegmentRef,
    },
};

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct MetadataRef {
    pub segment_id: SegmentId,
    pub bytes_offset: u64,
    pub bytes_length: u64,
    pub checksum: Checksum,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CollectionDesc {
    pub collection_id: CollectionId,
    pub name: String,
    pub created_at: i64,
    pub default_text_space_id: Option<TextSpaceId>,
    pub default_embedding_space_id: Option<EmbeddingSpaceId>,
    pub flags: u32,
    pub metadata_ref: Option<MetadataRef>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum FrameRole {
    Document,
    Chunk,
    Tombstone,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum FrameStatus {
    Active,
    Tombstoned,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum PayloadEncoding {
    Raw,
    Zstd,
    Lz4,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FrameDesc {
    pub frame_id: FrameId,
    pub collection_id: CollectionId,
    pub role: FrameRole,
    pub created_at: i64,
    pub updated_at: i64,
    pub uri_ref: Option<MetadataRef>,
    pub title_ref: Option<MetadataRef>,
    pub payload_ref: Option<PayloadRef>,
    pub payload_encoding: PayloadEncoding,
    pub search_text_ref: Option<MetadataRef>,
    pub parent_frame_id: Option<FrameId>,
    pub chunk_index: Option<u32>,
    pub chunk_count: Option<u32>,
    pub metadata_ref: Option<MetadataRef>,
    pub status: FrameStatus,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct TextSpaceDesc {
    pub text_space_id: TextSpaceId,
    pub name: String,
    pub analyzer_id: AnalyzerId,
    pub field_schema_id: FieldSchemaId,
    pub default_profile_id: RetrievalProfileId,
    pub flags: u32,
    pub enabled_retrievers: u32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum UnicodeNormalization {
    None,
    Nfc,
    Nfkc,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Tokenizer {
    UnicodeWords,
    Whitespace,
    CharNgram,
    TokenShingle,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum Stemming {
    None,
    /// Snowball / Porter2 English stemmer. Matches Lucene's
    /// `EnglishAnalyzer` and Anserini/pyserini defaults — what the BEIR
    /// paper's BM25 baseline reports.
    Porter2English,
}

/// Built-in stopword sets selected by [`AnalyzerDesc::stopwords`].
///
/// The existing `stopword_set_ref` field is reserved for user-supplied
/// stopword lists carried as Metadata blobs; this enum is the simpler
/// "pick a canned set" knob used by the BEIR-standard analyzer.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum StopwordsPolicy {
    None,
    /// The 33-word English stopword list shipped with Lucene's
    /// `EnglishAnalyzer` — `a, an, and, are, as, at, be, but, by, for,
    /// if, in, into, is, it, no, not, of, on, or, such, that, the,
    /// their, then, there, these, they, this, to, was, will, with`.
    /// Anserini/pyserini use this set unmodified, so it's the
    /// apples-to-apples choice for BEIR comparisons.
    EnglishLucene,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum PunctuationPolicy {
    Drop,
    Keep,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct AnalyzerDesc {
    pub analyzer_id: AnalyzerId,
    pub unicode_normalization: UnicodeNormalization,
    pub case_fold: bool,
    pub accent_fold: bool,
    pub tokenizer: Tokenizer,
    pub stemming: Stemming,
    pub stopword_set_ref: Option<MetadataRef>,
    /// When true, the stopword set is applied at query time only and the
    /// canonical postings still include stopword tokens (the v1 default;
    /// see spec §10.4).
    pub stopword_query_only: bool,
    /// Built-in stopword set applied inside `crate::text::analyzer::Analyzer`
    /// at analyze time (both indexing and query). Independent of the
    /// reference-resolved `stopword_set_ref` path; left as
    /// [`StopwordsPolicy::None`] for the legacy Valise defaults and set to
    /// [`StopwordsPolicy::EnglishLucene`] for BEIR-comparable runs.
    pub stopwords: StopwordsPolicy,
    /// Strip a trailing English possessive `'s` (or `'s` with curly
    /// quote U+2019) from each token, matching Lucene's
    /// `EnglishPossessiveFilter`. Off by default; on for the
    /// BEIR-comparable pipeline.
    pub english_possessive_strip: bool,
    /// Tokens shorter than this length (in characters) are dropped. `0`
    /// means no lower bound.
    pub min_token_len: u16,
    /// Tokens longer than this length (in characters) are dropped. `0`
    /// means no upper bound.
    pub max_token_len: u16,
    pub shingle_size: u8,
    pub ngram_min: Option<u8>,
    pub ngram_max: Option<u8>,
    pub punctuation_policy: PunctuationPolicy,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FieldSchemaDesc {
    pub field_schema_id: FieldSchemaId,
    pub fields: Vec<FieldDesc>,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FieldDesc {
    pub field_id: u16,
    pub name: String,
    pub source: FieldSource,
    pub store_positions: bool,
    pub store_term_freq: bool,
    pub store_set_membership: bool,
    pub weight: f32,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum FieldSource {
    SearchText,
    Title,
    Uri,
    Tags,
    MetadataPath(String),
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct RetrievalProfileDesc {
    pub profile_id: RetrievalProfileId,
    pub profile_type: RetrievalProfileType,
    pub params: RetrievalProfileParams,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum RetrievalProfileType {
    Bm25,
    Tfidf,
    JaccardExact,
    JaccardWeighted,
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum RetrievalProfileParams {
    Bm25 {
        k1: f32,
        b: f32,
        idf_variant: IdfVariant,
    },
    Tfidf {
        tf_mode: TfMode,
        idf_mode: IdfMode,
        norm_mode: NormMode,
    },
    JaccardExact {
        token_source: TokenSource,
    },
    JaccardWeighted {
        weight_source: WeightSource,
        token_source: TokenSource,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum IdfVariant {
    RobertsonSparckJones,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum TfMode {
    Raw,
    Log,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum IdfMode {
    Smooth,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum NormMode {
    None,
    L2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum TokenSource {
    Terms,
    Shingles,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum WeightSource {
    TermFrequency,
    Tfidf,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct EmbeddingSpaceDesc {
    pub embedding_space_id: EmbeddingSpaceId,
    pub provider: String,
    pub model: String,
    pub dimension: u32,
    pub metric: VectorMetric,
    pub normalized: bool,
    /// Input dtype for the space. Non-f8 spaces store `QamLloydMax(5,6)`
    /// codes via `primary_codec_id`; f8 spaces store raw bytes.
    pub dtype: crate::format::dtype::Dtype,
    /// Primary codec (`QamLloydMax` family). Required for non-f8 spaces;
    /// MUST be `None` for f8 spaces.
    pub primary_codec_id: Option<CodecId>,
    /// DEAD since v2.3 (vote-index removal). Always `None`; retained for
    /// format stability. `register_embedding_space` rejects non-`None`.
    pub secondary_codec_id: Option<CodecId>,
    /// Reserved flag word. No live consumer since the vote/ANN burial;
    /// always 0.
    pub flags: u32,
}

impl EmbeddingSpaceDesc {
    /// Resolve the codec used to encode/decode primary base bytes for
    /// this space. Returns `None` for f8 spaces (no codec — raw
    /// storage). `put_vector` and the search distance paths call this
    /// once per space.
    #[must_use]
    pub fn primary_codec(&self) -> Option<CodecId> {
        self.primary_codec_id
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum VectorMetric {
    Cosine,
    InnerProduct,
    L2,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CodecDesc {
    pub codec_id: CodecId,
    pub family: CodecFamily,
    pub version: u16,
    pub params_ref: SegmentRef,
    pub deterministic_id: Checksum,
}

impl CodecDesc {
    pub fn validate(&self) -> Result<()> {
        if self.params_ref.length == 0 {
            return Err(Error::Format(
                "CodecDesc.params_ref.length must be non-zero (CodecParamsSegment is mandatory)"
                    .into(),
            ));
        }
        if self.params_ref.checksum == [0u8; 32] {
            return Err(Error::Format(
                "CodecDesc.params_ref.checksum must be set".into(),
            ));
        }
        Ok(())
    }
}

/// Codec family discriminant.
///
/// - `QamLloydMax` is the production primary family — block-Hadamard
///   rotation + per-pair Lloyd-Max amplitude (5 bits) and phase (6 bits)
///   quantization. Attached to an embedding space's `primary_codec_id`.
/// - `_LegacyInt4PcaZero` is a retired v2.1 family, kept only as a bincode
///   discriminant placeholder (see the variant doc below).
///
/// Future minor versions may add sibling families (`ProductQuantized`,
/// `LearnedQuantized`, …); readers reject unknown discriminants per
/// spec §20.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum CodecFamily {
    /// Reserved placeholder — was `Int4PcaZero` in v2.1. The variant
    /// stays so bincode's positional discriminant for `QamLloydMax`
    /// remains `1`; existing on-disk codec descriptors keep loading.
    /// New code MUST NOT construct or match this variant; readers
    /// should treat it as an error when it appears.
    _LegacyInt4PcaZero,
    QamLloydMax,
    /// Unrestricted polar quantization (v2.4): one joint cell index
    /// per complex pair over `M` amplitude rings with power-of-two
    /// per-ring phase counts. Params blob: `format::upq_params`
    /// (`NXUP`). Appended at ordinal 2 — see the ordinal-stability
    /// note above.
    Upq,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum VectorStatus {
    Active,
    Tombstoned,
}

#[derive(Clone, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct VectorDesc {
    pub vector_id: VectorId,
    pub owner_frame_id: FrameId,
    pub collection_id: CollectionId,
    pub embedding_space_id: EmbeddingSpaceId,
    pub data_segment_id: SegmentId,
    pub ordinal_in_segment: u32,
    pub status: VectorStatus,
}

/// Score normalization for hybrid fusion, spec §18.3. `Rrf` ignores raw
/// score magnitudes and uses ranks only; `ZScore`/`MinMax` rescale per
/// channel before applying weights.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub enum FusionNormalization {
    None,
    ZScore,
    MinMax,
    /// Reciprocal Rank Fusion. `k` is the rank smoothing constant
    /// (typical: 60).
    Rrf {
        k: u32,
    },
}

#[derive(Clone, Debug, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FusionProfileDesc {
    pub fusion_profile_id: FusionProfileId,
    pub bm25_weight: f32,
    pub tfidf_weight: f32,
    pub jaccard_weight: f32,
    pub vector_weight: f32,
    pub normalization: FusionNormalization,
}

impl FusionProfileDesc {
    pub fn validate(&self) -> Result<()> {
        for (label, w) in [
            ("bm25_weight", self.bm25_weight),
            ("tfidf_weight", self.tfidf_weight),
            ("jaccard_weight", self.jaccard_weight),
            ("vector_weight", self.vector_weight),
        ] {
            if !w.is_finite() {
                return Err(Error::Format(format!(
                    "FusionProfileDesc {label} must be finite, got {w}"
                )));
            }
            if w < 0.0 {
                return Err(Error::Format(format!(
                    "FusionProfileDesc {label} must be non-negative, got {w}"
                )));
            }
        }
        let sum = self.bm25_weight + self.tfidf_weight + self.jaccard_weight + self.vector_weight;
        if sum <= 0.0 {
            return Err(Error::Format(
                "FusionProfileDesc requires at least one positive channel weight".into(),
            ));
        }
        if let FusionNormalization::Rrf { k } = self.normalization
            && k == 0
        {
            return Err(Error::Format("FusionProfileDesc RRF k must be >= 1".into()));
        }
        Ok(())
    }
}

/// Lightweight per-frame summary built at open without decoding the full
/// `FrameDesc`. Holds only the fields the retrieval hot path needs:
/// `frame_id` (to map back to corpus rows) and `status` (to filter
/// tombstoned frames). Heavy fields (`uri_ref`, `payload_ref`, etc.)
/// stay on disk and are only decoded when the user retrieves a top-K
/// result by id.
///
/// Built by the **Ghost Catalog** open-path: `read_frame_catalog_slim`
/// walks the catalog record envelopes (each is length-prefixed by spec),
/// reads `frame_id` as a leading varint, and reads `status` as the
/// trailing 1-byte enum discriminant — skipping the entire bincode
/// payload of each record. ~16 µs total for 200 k frames vs ~52 ms for
/// full bincode decode.
#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct FrameStub {
    pub frame_id: FrameId,
    pub status: FrameStatus,
}

#[derive(Clone, Debug, Default, PartialEq, serde::Deserialize, serde::Serialize)]
pub struct CatalogSnapshot {
    pub collections: Vec<CollectionDesc>,
    /// Heavy per-frame catalog. Left **empty** after the open fast-path —
    /// the retrieval-relevant `(frame_id, status)` pairs live in
    /// `frame_stubs` instead. Populated lazily by `frames_decoded()` (only
    /// when a public API needs heavy fields like `uri_ref` / `payload_ref`).
    pub frames: Vec<FrameDesc>,
    /// Slim frame index built at open by the Ghost-Catalog path.
    /// Sorted ascending by `frame_id`.
    pub frame_stubs: Vec<FrameStub>,
    pub text_spaces: Vec<TextSpaceDesc>,
    pub analyzers: Vec<AnalyzerDesc>,
    pub field_schemas: Vec<FieldSchemaDesc>,
    pub retrieval_profiles: Vec<RetrievalProfileDesc>,
    pub embedding_spaces: Vec<EmbeddingSpaceDesc>,
    pub codecs: Vec<CodecDesc>,
    pub vectors: Vec<VectorDesc>,
    pub fusion_profiles: Vec<FusionProfileDesc>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use bincode::serde::{decode_from_slice, encode_to_vec};

    fn well_formed_params_ref() -> SegmentRef {
        SegmentRef {
            segment_id: SegmentId(11),
            offset: 4096,
            length: 256,
            checksum: [7u8; 32],
        }
    }

    fn sample_codec_desc() -> CodecDesc {
        CodecDesc {
            codec_id: CodecId(3),
            family: CodecFamily::QamLloydMax,
            version: 1,
            params_ref: well_formed_params_ref(),
            deterministic_id: [9u8; 32],
        }
    }

    #[test]
    fn codec_family_bincode_roundtrip() {
        let bytes =
            encode_to_vec(CodecFamily::QamLloydMax, bincode::config::standard()).expect("encode");
        let (decoded, _): (CodecFamily, _) =
            decode_from_slice(&bytes, bincode::config::standard()).expect("decode");
        assert_eq!(decoded, CodecFamily::QamLloydMax);
    }

    fn sample_vector_desc() -> VectorDesc {
        VectorDesc {
            vector_id: VectorId(42),
            owner_frame_id: FrameId(7),
            collection_id: CollectionId(1),
            embedding_space_id: EmbeddingSpaceId(2),
            data_segment_id: SegmentId(99),
            ordinal_in_segment: 5,
            status: VectorStatus::Active,
        }
    }

    fn sample_analyzer_desc(tokenizer: Tokenizer) -> AnalyzerDesc {
        AnalyzerDesc {
            analyzer_id: AnalyzerId(1),
            unicode_normalization: UnicodeNormalization::Nfkc,
            case_fold: true,
            accent_fold: false,
            tokenizer,
            stemming: Stemming::None,
            stopword_set_ref: None,
            stopword_query_only: true,
            stopwords: StopwordsPolicy::None,
            english_possessive_strip: false,
            min_token_len: 2,
            max_token_len: 64,
            shingle_size: 0,
            ngram_min: None,
            ngram_max: None,
            punctuation_policy: PunctuationPolicy::Drop,
        }
    }

    #[test]
    fn analyzer_desc_bincode_roundtrip_each_tokenizer() {
        for tokenizer in [
            Tokenizer::UnicodeWords,
            Tokenizer::Whitespace,
            Tokenizer::CharNgram,
            Tokenizer::TokenShingle,
        ] {
            let desc = sample_analyzer_desc(tokenizer);
            let bytes =
                encode_to_vec(&desc, bincode::config::standard()).expect("encode AnalyzerDesc");
            let (decoded, _): (AnalyzerDesc, _) =
                decode_from_slice(&bytes, bincode::config::standard())
                    .expect("decode AnalyzerDesc");
            assert_eq!(decoded, desc);
        }
    }

    #[test]
    fn vector_desc_bincode_roundtrip() {
        let desc = sample_vector_desc();
        let bytes = encode_to_vec(&desc, bincode::config::standard()).expect("encode VectorDesc");
        let (decoded, _): (VectorDesc, _) =
            decode_from_slice(&bytes, bincode::config::standard()).expect("decode VectorDesc");
        assert_eq!(decoded, desc);
    }

    #[test]
    fn codec_family_rejects_unknown_discriminant() {
        // bincode 2 + serde encodes enum variants as a varint index.
        // Variant 0 is the reserved-legacy placeholder, 1 is
        // `QamLloydMax`, 2 is `Upq` (v2.4). Variant 3+ does not exist
        // and decoders must reject it so future families can't be
        // mis-interpreted by older readers.
        for future in [3u8, 4u8] {
            let bytes = [future];
            let err = decode_from_slice::<CodecFamily, _>(&bytes, bincode::config::standard())
                .expect_err("unknown discriminant should fail");
            let _ = err;
        }
    }

    #[test]
    fn codec_desc_bincode_roundtrip() {
        let desc = sample_codec_desc();
        let bytes = encode_to_vec(&desc, bincode::config::standard()).expect("encode");
        let (decoded, _): (CodecDesc, _) =
            decode_from_slice(&bytes, bincode::config::standard()).expect("decode");
        assert_eq!(decoded, desc);
    }

    #[test]
    fn codec_desc_validate_accepts_well_formed() {
        sample_codec_desc()
            .validate()
            .expect("well-formed should pass");
    }

    #[test]
    fn codec_desc_validate_rejects_zero_length() {
        let mut desc = sample_codec_desc();
        desc.params_ref.length = 0;
        let err = desc.validate().expect_err("zero length should fail");
        assert!(
            err.to_string().contains("length"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn codec_desc_validate_rejects_zero_checksum() {
        let mut desc = sample_codec_desc();
        desc.params_ref.checksum = [0u8; 32];
        let err = desc.validate().expect_err("zero checksum should fail");
        assert!(
            err.to_string().contains("checksum"),
            "unexpected error: {err}"
        );
    }

    fn sample_fusion_profile_desc() -> FusionProfileDesc {
        FusionProfileDesc {
            fusion_profile_id: FusionProfileId(11),
            bm25_weight: 0.5,
            tfidf_weight: 0.0,
            jaccard_weight: 0.0,
            vector_weight: 0.5,
            normalization: FusionNormalization::ZScore,
        }
    }

    #[test]
    fn fusion_profile_desc_bincode_roundtrip_each_normalization() {
        for normalization in [
            FusionNormalization::None,
            FusionNormalization::ZScore,
            FusionNormalization::MinMax,
            FusionNormalization::Rrf { k: 60 },
        ] {
            let mut desc = sample_fusion_profile_desc();
            desc.normalization = normalization;
            let bytes = encode_to_vec(&desc, bincode::config::standard()).expect("encode");
            let (decoded, _): (FusionProfileDesc, _) =
                decode_from_slice(&bytes, bincode::config::standard()).expect("decode");
            assert_eq!(decoded, desc);
        }
    }

    #[test]
    fn fusion_profile_desc_rejects_negative_weight() {
        let mut desc = sample_fusion_profile_desc();
        desc.bm25_weight = -0.1;
        let err = desc.validate().expect_err("negative weight should fail");
        assert!(err.to_string().contains("non-negative"));
    }

    #[test]
    fn fusion_profile_desc_rejects_nan_weight() {
        let mut desc = sample_fusion_profile_desc();
        desc.tfidf_weight = f32::NAN;
        let err = desc.validate().expect_err("NaN weight should fail");
        assert!(err.to_string().contains("finite"));
    }

    #[test]
    fn fusion_profile_desc_rejects_all_zero_weights() {
        let mut desc = sample_fusion_profile_desc();
        desc.bm25_weight = 0.0;
        desc.tfidf_weight = 0.0;
        desc.jaccard_weight = 0.0;
        desc.vector_weight = 0.0;
        let err = desc.validate().expect_err("all-zero weights should fail");
        assert!(err.to_string().contains("at least one positive"));
    }

    #[test]
    fn fusion_profile_desc_rejects_rrf_k_zero() {
        let mut desc = sample_fusion_profile_desc();
        desc.normalization = FusionNormalization::Rrf { k: 0 };
        let err = desc.validate().expect_err("RRF k=0 should fail");
        assert!(err.to_string().contains("RRF k"));
    }
}
