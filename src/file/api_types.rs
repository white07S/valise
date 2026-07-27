//! Public API value types for the `ValiseFile` engine surface: create/open
//! options, the `put_*` inputs, and the read-result/spec types. Re-exported
//! from `file.rs` so the public path stays `crate::file::<Type>`.

use crate::error::Result;
use crate::format::catalog::{FrameRole, VectorMetric};
use crate::format::dtype::{Dtype, DtypeSet};
use crate::format::{CodecId, CollectionId, EmbeddingSpaceId, FrameId};
use crate::io::Durability;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum OpenMode {
    ReadOnly,
    ReadWrite,
}

#[derive(Clone, Debug)]
pub struct CreateOptions {
    pub durability: Durability,
    /// Whether text retrieval is enabled for this file. When
    /// `TextMode::Disabled`, `register_text_space` and friends fail
    /// fast with `Error::Unsupported`. Captured at create time and
    /// frozen in the file-level contract — flipping it later requires
    /// rebuilding the file.
    pub text: TextMode,
    /// Hard upper bounds on the vector subsystem: the maximum
    /// embedding-space dimension this file will accept and the dtype
    /// whitelist that gates `register_embedding_space`.
    pub vector: VectorContract,
    /// VESTIGIAL since v2.3. These thresholds gated the now-removed CSR
    /// vote-index auto-build; vector search derives a sign-sketch at file-open
    /// instead, so nothing reads them at runtime. Persisted in the create
    /// contract for format stability only. See docs/VECTOR_SEARCH.md.
    pub auto_promote: AutoPromote,
}

impl Default for CreateOptions {
    fn default() -> Self {
        Self {
            durability: Durability::default(),
            text: TextMode::Enabled,
            vector: VectorContract::default(),
            auto_promote: AutoPromote::default(),
        }
    }
}

/// Whether text retrieval is enabled for a file. Pinned at create
/// time; read-only thereafter via [`ValiseFile::text_enabled`](crate::ValiseFile::text_enabled).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TextMode {
    Enabled,
    Disabled,
}

/// File-level vector contract.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct VectorContract {
    /// Maximum embedding-space dimension. Embedding spaces with
    /// `dimension > max_dim` are rejected at registration. 1..=65_535.
    pub max_dim: u32,
    /// Allowed dtypes for `register_embedding_space`. Default:
    /// [`DtypeSet::ALL`].
    pub allowed_dtypes: DtypeSet,
}

impl Default for VectorContract {
    fn default() -> Self {
        Self {
            max_dim: 4096,
            allowed_dtypes: DtypeSet::ALL,
        }
    }
}

/// Auto-promotion thresholds. VESTIGIAL since v2.3: these gated the removed
/// CSR vote-index auto-build. No runtime path reads them now — vector search
/// derives an in-memory sign-sketch at file-open regardless of corpus size.
/// Kept (and still validated by the create contract) only so existing files
/// and the public option struct stay layout-stable. See docs/VECTOR_SEARCH.md.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoPromote {
    /// Active-vector count at which non-f8 spaces auto-promote.
    /// Default: 200_000.
    pub non_f8_threshold: u64,
    /// Active-vector count at which f8 spaces auto-promote. MUST be ≥
    /// `non_f8_threshold`. Default: 500_000.
    pub f8_threshold: u64,
}

impl Default for AutoPromote {
    fn default() -> Self {
        Self {
            non_f8_threshold: 200_000,
            f8_threshold: 500_000,
        }
    }
}

impl CreateOptions {
    /// Lower the user-facing options into the persisted contract record.
    /// Validation runs on the result; the caller is expected to surface
    /// the error before any bytes hit the disk.
    pub(crate) fn build_contract(
        &self,
    ) -> Result<crate::format::create_contract::CreateContractV1> {
        let contract = crate::format::create_contract::CreateContractV1 {
            schema_version: crate::format::create_contract::SCHEMA_VERSION,
            text_enabled: matches!(self.text, TextMode::Enabled),
            max_dim: self.vector.max_dim,
            allowed_dtypes: self.vector.allowed_dtypes.bits(),
            auto_promote_non_f8: self.auto_promote.non_f8_threshold,
            auto_promote_f8: self.auto_promote.f8_threshold,
            reserved: [0u8; 16],
        };
        contract.validate_options()?;
        Ok(contract)
    }
}

/// Input to [`ValiseFile::put_frame`](crate::ValiseFile::put_frame). The payload is borrowed for the
/// duration of the call — `put_frame` writes it synchronously to a
/// `Payload` segment and copies whatever metadata it needs into its own
/// in-memory state. Callers can pass `&[u8]` (string literals, slices
/// of an Arrow column, etc.) without an owning copy.
#[derive(Clone, Debug)]
pub struct PutFrame<'a> {
    pub collection_id: CollectionId,
    pub role: FrameRole,
    pub payload: &'a [u8],
    pub created_at: Option<i64>,
    pub updated_at: Option<i64>,
    pub parent_frame_id: Option<FrameId>,
    pub chunk_index: Option<u32>,
    pub chunk_count: Option<u32>,
    /// Optional external identity key / URI for this frame. When set, the
    /// bytes are persisted in-file via `FrameDesc.uri_ref` (a
    /// `MetadataRef` into a batched `Metadata` segment), letting a higher
    /// layer rebuild a stable key → frame mapping at open without
    /// sidecars. `None` for anonymous frames (the legacy behavior).
    pub uri: Option<&'a [u8]>,
}

impl<'a> PutFrame<'a> {
    #[must_use]
    pub fn document(collection_id: CollectionId, payload: &'a [u8]) -> Self {
        Self {
            collection_id,
            role: FrameRole::Document,
            payload,
            created_at: None,
            updated_at: None,
            parent_frame_id: None,
            chunk_index: None,
            chunk_count: None,
            uri: None,
        }
    }

    /// Attach an external identity key/URI, persisted via `uri_ref`.
    #[must_use]
    pub fn with_uri(mut self, uri: &'a [u8]) -> Self {
        self.uri = Some(uri);
        self
    }
}

/// Input to [`ValiseFile::put_vector`](crate::ValiseFile::put_vector). `values` is borrowed for the
/// duration of the call — `put_vector` `extend_from_slice`s the floats
/// into a single contiguous per-batch buffer (one allocation amortized
/// across the whole batch, regardless of how many `put_vector` calls
/// the user makes).
#[derive(Clone, Debug)]
pub struct PutVector<'a> {
    pub owner_frame_id: FrameId,
    pub embedding_space_id: EmbeddingSpaceId,
    pub values: &'a [f32],
}

/// Selects the score path used by `vector_search` (and the deprecated
/// `vector_search`). `Lossy` keeps the quantized asymmetric-distance
/// scores from the index; `Full` reranks the post-filter top hits by
/// decoding each candidate to raw f32 and recomputing the metric on
/// the dequantized vectors.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VectorFidelity {
    Lossy,
    Full,
}

/// `read_vector` representation selector (plan §11).
///
/// - [`Reconstruct::StoredBytes`] returns the codec base bytes
///   verbatim — copy-out of the mmap, no decode work.
/// - [`Reconstruct::F32Vector`] decodes through the primary codec and
///   returns a `Vec<f32>` of length `space.dimension`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reconstruct {
    StoredBytes,
    F32Vector,
}

/// Output of [`ValiseFile::read_vector`](crate::ValiseFile::read_vector). The variant matches the
/// [`Reconstruct`] passed in.
#[derive(Clone, Debug)]
pub enum ReadVectorResult {
    StoredBytes(Vec<u8>),
    F32Vector(Vec<f32>),
}

impl ReadVectorResult {
    /// Project to `Vec<f32>`, panicking if the variant is
    /// `StoredBytes`. Convenience for callers that already passed
    /// `Reconstruct::F32Vector`.
    #[must_use]
    pub fn expect_f32(self) -> Vec<f32> {
        match self {
            Self::F32Vector(v) => v,
            Self::StoredBytes(_) => {
                panic!("ReadVectorResult::expect_f32: got StoredBytes variant")
            }
        }
    }

    /// Project to `Vec<u8>`, panicking if the variant is `F32Vector`.
    /// Convenience for callers that already passed
    /// `Reconstruct::StoredBytes`.
    #[must_use]
    pub fn expect_stored_bytes(self) -> Vec<u8> {
        match self {
            Self::StoredBytes(b) => b,
            Self::F32Vector(_) => {
                panic!("ReadVectorResult::expect_stored_bytes: got F32Vector variant")
            }
        }
    }
}

/// User-facing spec for `register_embedding_space`. Carries a primary
/// codec (`QamLloydMax`) and an explicit input `dtype`. Validation rules
/// are enforced at register time and surface as
/// `Error::Format` / `Error::Unsupported`.
#[derive(Clone, Debug)]
pub struct EmbeddingSpaceSpec {
    pub provider: String,
    pub model: String,
    pub dimension: u32,
    pub metric: VectorMetric,
    pub normalized: bool,
    /// Input precision for vectors that land in this space. Defaults
    /// to [`Dtype::F32`].
    pub dtype: Dtype,
    /// Primary codec (`QamLloydMax` family). Required when
    /// `!dtype.is_f8()`; MUST be `None` for f8 spaces.
    pub primary_codec_id: Option<CodecId>,
    /// DEPRECATED / unused since v2.3. Was an optional secondary QAM codec
    /// consumed by the removed CSR vote-index build. `register_embedding_space`
    /// now rejects any non-`None` value. Leave it `None`.
    pub secondary_codec_id: Option<CodecId>,
}

impl Default for EmbeddingSpaceSpec {
    fn default() -> Self {
        Self {
            provider: String::new(),
            model: String::new(),
            dimension: 0,
            metric: VectorMetric::Cosine,
            normalized: false,
            dtype: Dtype::F32,
            primary_codec_id: None,
            secondary_codec_id: None,
        }
    }
}
