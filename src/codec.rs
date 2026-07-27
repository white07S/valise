// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Vector codecs (spec §14). Two production families: `QamLloydMax`
//! (Lloyd-Max-trained polar codebook; `QamSliding` is its fast NEON
//! scoring kernel) and `Upq` (unrestricted polar quantization, v2.4).
//!
//! Family dispatch is centralized in [`CodecParams`]: persisting,
//! restoring, fingerprinting, and instantiating a codec all route
//! through its match arms, so adding a family means adding one variant
//! here plus the search-path wiring the compiler then surfaces.

pub(crate) mod prng;
pub(crate) mod qam_lloyd_max;
pub(crate) mod qam_sliding;
pub(crate) mod upq;

use crate::{
    error::{Error, Result},
    format::catalog::{CodecFamily, VectorMetric},
};

pub use crate::format::qam_lloyd_max_params::QamLloydMaxParams;
pub use crate::format::upq_params::{UpqDesignSource, UpqParams};

/// Family-tagged codec parameter set — the single dispatch point between
/// the persisted `CodecParams` segment and a live `VectorCodec`.
///
/// Every per-family persisted detail (wire codec, wire version, stable
/// deterministic-id discriminant, instantiation) is answered by a match
/// on this enum. The discriminants and encodings are part of the on-disk
/// format; see `docs/FORMAT.md` §14 and `MIGRATION.md`.
#[derive(Clone, Debug, PartialEq)]
pub enum CodecParams {
    QamLloydMax(QamLloydMaxParams),
    Upq(UpqParams),
}

impl CodecParams {
    /// Catalog family tag persisted in `CodecDesc.family`.
    pub fn family(&self) -> CodecFamily {
        match self {
            CodecParams::QamLloydMax(_) => CodecFamily::QamLloydMax,
            CodecParams::Upq(_) => CodecFamily::Upq,
        }
    }

    /// Per-family params wire version persisted in `CodecDesc.version`.
    /// Bump only with a wire-format change to that family's params blob.
    pub(crate) fn wire_version(&self) -> u16 {
        match self {
            CodecParams::QamLloydMax(_) => 1,
            CodecParams::Upq(_) => 1,
        }
    }

    /// Stable wire discriminant feeding the content-addressed
    /// `deterministic_id`. Append-only: existing values MUST NOT change
    /// or persisted ids drift (`_LegacyInt4PcaZero` holds 0).
    pub(crate) fn family_discriminant(&self) -> u32 {
        match self {
            CodecParams::QamLloydMax(_) => 1,
            CodecParams::Upq(_) => 2,
        }
    }

    /// Vector dimension this codec was fit for.
    pub fn dimension(&self) -> u32 {
        match self {
            CodecParams::QamLloydMax(p) => p.dimension,
            CodecParams::Upq(p) => p.dimension,
        }
    }

    /// Encode to the family's params blob (magic + version + body).
    pub(crate) fn encode(&self) -> Result<Vec<u8>> {
        match self {
            CodecParams::QamLloydMax(p) => {
                crate::format::qam_lloyd_max_params::encode_qam_lloyd_max_params(p)
            }
            CodecParams::Upq(p) => crate::format::upq_params::encode_upq_params(p),
        }
    }

    /// Decode a params blob according to the catalog family tag.
    pub(crate) fn decode(family: CodecFamily, bytes: &[u8]) -> Result<Self> {
        match family {
            CodecFamily::QamLloydMax => Ok(CodecParams::QamLloydMax(
                crate::format::qam_lloyd_max_params::decode_qam_lloyd_max_params(bytes)?,
            )),
            CodecFamily::Upq => Ok(CodecParams::Upq(
                crate::format::upq_params::decode_upq_params(bytes)?,
            )),
            CodecFamily::_LegacyInt4PcaZero => Err(Error::Unsupported(
                "CodecParams::decode: the Int4PcaZero family was removed in v2.2; \
                 rebuild the .vls with a current codec"
                    .into(),
            )),
        }
    }

    /// Instantiate the live codec for these params.
    pub(crate) fn instantiate(&self) -> Result<Box<dyn VectorCodec>> {
        Ok(match self {
            CodecParams::QamLloydMax(p) => {
                Box::new(qam_lloyd_max::QamLloydMaxCodec::from_params(p)?)
            }
            CodecParams::Upq(p) => Box::new(upq::UpqCodec::from_params(p)?),
        })
    }
}

/// Bench-only entry points for the per-kernel SIMD functions so the
/// criterion benches in `benches/` can time each kernel against its
/// scalar reference without going through the codec wrapper. Gated
/// behind the `bench` cargo feature.
///
/// These are thin `pub fn` wrappers (not `pub use`) because the
/// underlying kernels are `pub(crate)`; re-exporting them as `pub`
/// would violate the visibility chain. The wrappers are
/// `#[inline(always)]` so there's no overhead vs a direct call.
#[cfg(any(test, feature = "bench"))]
pub mod simd_bench {
    use crate::codec::qam_lloyd_max::simd;

    #[inline(always)]
    pub fn asymmetric_dot_pairs(
        q_rot: &[f32],
        amp_indices: &[u32],
        phase_indices: &[u32],
        sigma_per_pair: &[f32],
        amp_levels_unit: &[f32],
        phase_cos_lut: &[f32],
        phase_sin_lut: &[f32],
    ) -> (f32, f32) {
        simd::asymmetric_dot_pairs(
            q_rot,
            amp_indices,
            phase_indices,
            sigma_per_pair,
            amp_levels_unit,
            phase_cos_lut,
            phase_sin_lut,
        )
    }

    #[inline(always)]
    pub fn scalar_asymmetric_dot_pairs(
        q_rot: &[f32],
        amp_indices: &[u32],
        phase_indices: &[u32],
        sigma_per_pair: &[f32],
        amp_levels_unit: &[f32],
        phase_cos_lut: &[f32],
        phase_sin_lut: &[f32],
    ) -> (f32, f32) {
        simd::scalar_asymmetric_dot_pairs(
            q_rot,
            amp_indices,
            phase_indices,
            sigma_per_pair,
            amp_levels_unit,
            phase_cos_lut,
            phase_sin_lut,
        )
    }

    #[inline(always)]
    pub fn block_hadamard_forward(data: &mut [f32], signs: &[f32], block_size: usize) {
        simd::block_hadamard_forward(data, signs, block_size)
    }

    #[inline(always)]
    pub fn scalar_block_hadamard_forward(data: &mut [f32], signs: &[f32], block_size: usize) {
        simd::scalar_block_hadamard_forward(data, signs, block_size)
    }

    #[inline(always)]
    pub fn block_hadamard_inverse(data: &mut [f32], signs: &[f32], block_size: usize) {
        simd::block_hadamard_inverse(data, signs, block_size)
    }

    #[inline(always)]
    pub fn scalar_block_hadamard_inverse(data: &mut [f32], signs: &[f32], block_size: usize) {
        simd::scalar_block_hadamard_inverse(data, signs, block_size)
    }

    #[inline(always)]
    pub fn pair_magnitudes(pairs: &[f32], mags: &mut [f32]) {
        simd::pair_magnitudes(pairs, mags)
    }

    #[inline(always)]
    pub fn scalar_pair_magnitudes(pairs: &[f32], mags: &mut [f32]) {
        simd::scalar_pair_magnitudes(pairs, mags)
    }
}

/// **Bench-only adapter** around QAM Lloyd-Max. Gated behind the
/// `bench` feature; production callers go through
/// [`crate::ValiseFile::register_codec_qam`].
#[cfg(any(test, feature = "bench"))]
pub struct QamLloydMaxBench {
    inner: qam_lloyd_max::QamLloydMaxCodec,
}

#[cfg(any(test, feature = "bench"))]
impl QamLloydMaxBench {
    pub fn new(dim: usize) -> Result<Self> {
        Ok(Self {
            inner: qam_lloyd_max::QamLloydMaxCodec::new(dim)?,
        })
    }

    /// Construct with explicit `(block_size, amp_bits, phase_bits,
    /// renormalize_at_decode)`. The production config is `(1024, 5, 6,
    /// true)`.
    pub fn with_config(
        dim: usize,
        block_size: usize,
        amp_bits: u8,
        phase_bits: u8,
        renormalize_at_decode: bool,
    ) -> Result<Self> {
        Ok(Self {
            inner: qam_lloyd_max::QamLloydMaxCodec::with_config(
                dim,
                block_size,
                amp_bits,
                phase_bits,
                renormalize_at_decode,
            )?,
        })
    }

    pub fn calibrate(&mut self, sample: &[Vec<f32>]) -> Result<()> {
        self.inner.calibrate(sample)
    }

    pub fn to_params(&self, calibration_id: crate::format::Checksum) -> QamLloydMaxParams {
        self.inner.to_params(calibration_id)
    }

    pub fn dim(&self) -> usize {
        self.inner.dim
    }

    pub fn num_pairs(&self) -> usize {
        self.inner.num_pairs
    }

    pub fn amp_bits(&self) -> u8 {
        self.inner.amp_bits
    }

    pub fn phase_bits(&self) -> u8 {
        self.inner.phase_bits
    }

    pub fn renormalize_at_decode(&self) -> bool {
        self.inner.renormalize_at_decode
    }

    pub fn base_bytes_per_vector(&self) -> usize {
        self.inner.base_bytes_per_vector
    }

    pub fn query_bytes_per_vector(&self) -> usize {
        self.inner.query_bytes_per_vector
    }

    pub fn encode(&self, vector: &[f32]) -> Result<Vec<u8>> {
        self.inner.encode(vector)
    }

    pub fn decode_lossy(&self, base: &[u8]) -> Result<Vec<f32>> {
        self.inner.decode_lossy(base)
    }

    pub fn symmetric_distance(&self, a: &[u8], b: &[u8]) -> Result<f32> {
        self.inner
            .symmetric_distance(a, b, crate::format::catalog::VectorMetric::Cosine)
    }

    pub fn prepare_query(&self, query: &[f32]) -> Result<Vec<u8>> {
        self.inner.prepare_query_blob(query)
    }

    /// Bench helper: the forward-rotated vector (sign-LSH sketch input).
    /// Same block-Hadamard rotation the encoder applies, so DB and query
    /// sketches live in the same rotated space.
    pub fn rotate(&self, vector: &[f32]) -> Result<Vec<f32>> {
        Ok(self.inner.prepare_query_rotated(vector)?.0)
    }

    pub fn asymmetric_distance_prepared(&self, q_blob: &[u8], base: &[u8]) -> Result<f32> {
        self.inner.asymmetric_distance_prepared_blob(
            q_blob,
            base,
            crate::format::catalog::VectorMetric::Cosine,
        )
    }

    /// Bench helper: the register-LUT NEON sliding engine (ADC kernel) that
    /// scores the COMPACT QAM base bytes directly (5,6 only).
    pub fn sliding(&self) -> Result<SlidingBench> {
        Ok(SlidingBench {
            eng: qam_sliding::QamSlidingEngine::from_codec(&self.inner)?,
        })
    }
}

/// Bench handle around the register-LUT NEON sliding (ADC) kernel.
#[cfg(any(test, feature = "bench"))]
pub struct SlidingBench {
    eng: qam_sliding::QamSlidingEngine,
}
/// Prepared (i8-quantized) query for [`SlidingBench`].
#[cfg(any(test, feature = "bench"))]
pub struct SlidingPrep {
    prep: qam_sliding::QamSlidingPreparedQuery,
}
#[cfg(any(test, feature = "bench"))]
impl SlidingBench {
    pub fn prepare(&self, query: &[f32]) -> Result<SlidingPrep> {
        Ok(SlidingPrep {
            prep: self.eng.prepare_query(query)?,
        })
    }
    /// Per-row ‖ŷ‖ (cache at ingest for the hot path).
    pub fn y_hat_norm(&self, base: &[u8]) -> f32 {
        self.eng.y_hat_norm(base)
    }
    /// Hot path: score compact base bytes with a pre-cached row norm.
    pub fn score_with_norm(&self, p: &SlidingPrep, base: &[u8], y_hat_norm: f32) -> Result<f32> {
        self.eng
            .asymmetric_distance_with_norm(&p.prep, base, y_hat_norm)
    }
    /// Bench-only: integer SDOT kernel result. Routes through the
    /// arch dispatcher (aarch64 → NEON, otherwise → scalar). Useful
    /// for timing the kernel in isolation against
    /// [`Self::raw_dot_int_scalar`].
    pub fn raw_dot_int(&self, p: &SlidingPrep, base: &[u8]) -> i64 {
        self.eng.raw_dot_int(&p.prep, base)
    }
    /// Bench-only: scalar reference for the i8 SDOT kernel. Bit-
    /// comparable to the NEON path (the scalar reference mirrors
    /// `vqshrn_n_s16`'s rounding-saturating narrow).
    pub fn raw_dot_int_scalar(&self, p: &SlidingPrep, base: &[u8]) -> i64 {
        let amp_stream = &base
            [self.eng.amp_stream_offset..self.eng.amp_stream_offset + self.eng.amp_stream_bytes];
        let phase_stream = &base[self.eng.phase_stream_offset
            ..self.eng.phase_stream_offset + self.eng.phase_stream_bytes];
        self.eng
            .raw_dot_int_scalar(amp_stream, phase_stream, &p.prep.q_i8)
    }
}

/// Codec contract for the vector layer. Implementations must be deterministic
/// (spec §14.8): same params plus same input must produce byte-identical
/// encoded base bytes across runs and architectures.
///
/// Distance methods take the metric as a parameter rather than carrying it on
/// the codec, because metric is a per-embedding-space property (§10.7) and
/// one codec can be shared across multiple embedding spaces with different
/// metrics. Distances follow the "smaller = more similar" convention:
///
/// - `Cosine`: `1 - cos_sim`
/// - `InnerProduct`: `-ip`
/// - `L2`: squared L2 distance
pub(crate) trait VectorCodec: Send + Sync {
    /// Concrete-type access: returns `&dyn Any` so callers can
    /// `downcast_ref` to the underlying codec (e.g. `QamLloydMaxCodec`)
    /// for family-specific fast paths.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Catalog family tag for this codec instance.
    fn family(&self) -> CodecFamily;

    fn dimension(&self) -> u32;

    fn base_bytes_per_vector(&self) -> usize;

    /// Derive the 1-bit-per-dim sign sketch of a stored vector's rotated
    /// reconstruction straight from its packed base bytes — the
    /// candidate-generation input built once at file open. Must bit-match
    /// the query-side `retrieval::sketch::pack_query_sketch` packing.
    fn sign_sketch(&self, base: &[u8]) -> Result<Vec<u64>>;

    fn encode(&self, values: &[f32]) -> Result<Vec<u8>>;
    fn decode_lossy(&self, base: &[u8]) -> Result<Vec<f32>>;

    fn asymmetric_distance(&self, query: &[f32], base: &[u8], metric: VectorMetric) -> Result<f32>;

    /// Build an opaque per-query context that hoists rotation, validation,
    /// and norm computation out of the per-vector inner loop. Hot-path
    /// callers (the rerank / brute-force scan) call this once per query, then
    /// call `asymmetric_distance_prepared` for every scored vector.
    ///
    /// The returned `Box<dyn Any>` carries a codec-specific concrete type;
    /// pass it back to `asymmetric_distance_prepared` on the same codec
    /// instance. Default implementation stores the raw query and routes
    /// `asymmetric_distance_prepared` back through `asymmetric_distance` —
    /// correct but pays the per-call setup cost.
    fn prepare_query(&self, query: &[f32]) -> Result<Box<dyn std::any::Any + Send + Sync>> {
        Ok(Box::new(query.to_vec()))
    }

    /// Per-vector asymmetric distance using a pre-built query context.
    /// Codecs that override `prepare_query` MUST also override this to
    /// downcast to their own context type. Default: expects a `Vec<f32>`
    /// in the box and routes through the legacy path.
    fn asymmetric_distance_prepared(
        &self,
        ctx: &(dyn std::any::Any + Send + Sync),
        base: &[u8],
        metric: VectorMetric,
    ) -> Result<f32> {
        let query = ctx.downcast_ref::<Vec<f32>>().ok_or_else(|| {
            crate::error::Error::Integrity(
                "asymmetric_distance_prepared: default impl expects Vec<f32> ctx".into(),
            )
        })?;
        self.asymmetric_distance(query, base, metric)
    }
}
