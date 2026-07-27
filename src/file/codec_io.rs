//! Codec registration, persisted-params readback, and the open-time
//! codec cache — the engine side of the codec-family dispatch that
//! [`crate::codec::CodecParams`] centralizes.
//!
//! Adding a codec family touches: one `CodecParams` variant (and its
//! match arms), the family's params module in `src/format/`, the codec
//! implementation in `src/codec/`, and — only if the family needs a
//! family-specific rerank kernel — a search path next to
//! `file/upq_search.rs`. The registration plumbing below is shared.

use std::collections::HashMap;
use std::fs::File;

use super::segment_io::{
    SegmentRegistryMut, append_segment_at_end, read_segment_payload_typed, register_segment,
    segment_catalog_entry, segment_ref_at_end,
};
use super::{ValiseFile, calibration_id_from_sample, upsert_codec};
use crate::codec::{CodecParams, VectorCodec};
use crate::error::{Error, Result};
use crate::format::catalog::CodecDesc;
use crate::format::segment::SegmentType;
use crate::format::{CodecId, qam_lloyd_max_params::QamLloydMaxParams, upq_params};
use crate::io::sync_file;

impl ValiseFile {
    /// Register a vector codec from explicit, family-tagged parameters.
    ///
    /// Crash-safe order: params segment at EOF → sync → catalog upsert →
    /// live cache. The codec's content-addressed `deterministic_id` is
    /// BLAKE3 over `(family discriminant, wire version, params bytes)`.
    pub fn register_codec(&mut self, params: CodecParams) -> Result<CodecId> {
        self.ensure_write()?;

        let params_bytes = params.encode()?;
        let segment_id = self.id_allocator.allocate_segment_id()?;
        let params_segment = segment_ref_at_end(
            &self.file.lock(),
            segment_id,
            SegmentType::CodecParams,
            1,
            &params_bytes,
        )?;
        let deterministic_id = compute_codec_deterministic_id(&params, &params_bytes);
        let codec_id = self.id_allocator.allocate_codec_id()?;
        let codec_desc = CodecDesc {
            codec_id,
            family: params.family(),
            version: params.wire_version(),
            params_ref: params_segment,
            deterministic_id,
        };
        codec_desc.validate()?;

        let written_segment = append_segment_at_end(
            &mut self.file.lock(),
            segment_id,
            SegmentType::CodecParams,
            1,
            &params_bytes,
        )?;
        debug_assert_eq!(written_segment, params_segment);
        sync_file(&self.file.lock(), self.durability)?;

        register_segment(
            SegmentRegistryMut {
                registry: self.segment_registry.get_mut(),
                dirty_ids: &mut self.dirty_segment_ids,
            },
            segment_catalog_entry(
                segment_id,
                SegmentType::CodecParams,
                1,
                &params_bytes,
                params_segment,
            )?,
        );

        upsert_codec(&mut self.catalog, codec_desc);
        self.dirty_codec_ids.insert(codec_id);
        self.dirty = true;

        self.codec_cache.insert(codec_id, params.instantiate()?);

        Ok(codec_id)
    }

    /// Register a `QamLloydMax` codec from explicit parameters.
    pub fn register_codec_qam(&mut self, params: QamLloydMaxParams) -> Result<CodecId> {
        self.register_codec(CodecParams::QamLloydMax(params))
    }

    /// Production calibration entry: fit a `QamLloydMax` codec from a
    /// representative `sample` of vectors, then register it.
    ///
    /// [`Self::register_codec_qam`] requires a prebuilt
    /// `QamLloydMaxParams`, but those params can only be produced by
    /// fitting per-pair Rayleigh σ from real data — there is otherwise no
    /// production path to calibrate a codec, so vector/hybrid spaces can't
    /// be created without this. Fits σ via `QamLloydMaxCodec::calibrate`
    /// using the production amp/phase bits and a `block_size` chosen valid
    /// for `dim`, then hands the resulting params to
    /// [`Self::register_codec`] (which owns the crash-safe segment write +
    /// catalog upsert).
    ///
    /// `calibration_id` is a BLAKE3 digest over `(dim, sample)` so the
    /// same calibration data deterministically yields the same id
    /// (provenance tag; distinct from the codec's content-addressed
    /// `deterministic_id`, which the inner register computes).
    ///
    /// The sample must be non-empty and every vector must have length
    /// `dim` and be finite (validated inside `calibrate`).
    pub fn register_codec_qam_from_sample(
        &mut self,
        dim: usize,
        sample: &[Vec<f32>],
    ) -> Result<CodecId> {
        self.register_codec_qam_from_sample_with_bits(
            dim,
            crate::codec::qam_lloyd_max::DEFAULT_AMP_BITS,
            crate::codec::qam_lloyd_max::DEFAULT_PHASE_BITS,
            sample,
        )
    }

    /// Calibrate and register a QAM codec at an explicit `(amp_bits,
    /// phase_bits)` budget instead of the production `(5, 6)` default. Higher
    /// bits trade storage for recall — e.g. `(8, 8)` roughly doubles the
    /// codes but lifts the reconstruction ceiling (see `docs/VECTOR_SEARCH.md`).
    /// Any registered config takes the sketch-then-rerank search path: `(5, 6)`
    /// uses the SIMD sliding kernel, others the general asymmetric rerank.
    pub fn register_codec_qam_from_sample_with_bits(
        &mut self,
        dim: usize,
        amp_bits: u8,
        phase_bits: u8,
        sample: &[Vec<f32>],
    ) -> Result<CodecId> {
        self.ensure_write()?;
        let mut codec = crate::codec::qam_lloyd_max::QamLloydMaxCodec::for_calibration_with_bits(
            dim, amp_bits, phase_bits,
        )?;
        codec.calibrate(sample)?;
        let calibration_id = calibration_id_from_sample(dim, sample);
        let params = codec.to_params(calibration_id);
        self.register_codec_qam(params)
    }

    /// Register a `Upq` (unrestricted polar) codec from explicit
    /// parameters.
    pub fn register_codec_upq(&mut self, params: upq_params::UpqParams) -> Result<CodecId> {
        self.register_codec(CodecParams::Upq(params))
    }

    /// Production calibration entry for UPQ: fit on a representative
    /// `sample` at the default operating point (2048 cells = 11
    /// bits/pair = 5.5 bits/dim, empirical ring design) and register.
    pub fn register_codec_upq_from_sample(
        &mut self,
        dim: usize,
        sample: &[Vec<f32>],
    ) -> Result<CodecId> {
        self.register_codec_upq_from_sample_with_options(
            dim,
            crate::codec::upq::DEFAULT_UPQ_CELLS,
            upq_params::UpqDesignSource::Empirical,
            sample,
        )
    }

    /// UPQ calibration with an explicit cell budget and design source.
    /// `Rayleigh` designs the rings on the analytic density (data-free
    /// beyond the per-pair σ fit — the QAM-equivalent calibration
    /// burden); `Empirical` fits them on the sample.
    pub fn register_codec_upq_from_sample_with_options(
        &mut self,
        dim: usize,
        cells: u32,
        source: upq_params::UpqDesignSource,
        sample: &[Vec<f32>],
    ) -> Result<CodecId> {
        let codec = crate::codec::upq::UpqCodec::fit_from_sample(
            dim,
            crate::codec::upq::default_block_size_for(dim),
            crate::codec::upq::DEFAULT_UPQ_ROTATION_SEED,
            cells,
            crate::codec::upq::DEFAULT_UPQ_RING_SWEEP,
            source,
            sample,
        )?;
        let calibration_id = calibration_id_from_sample(dim, sample);
        let params = codec.to_params(calibration_id);
        self.register_codec_upq(params)
    }

    /// Decode and return a registered codec's persisted parameters,
    /// family-tagged (compaction / lossless re-encode hook).
    pub fn codec_params(&self, codec_id: CodecId) -> Result<CodecParams> {
        let codec = self
            .catalog
            .codecs
            .iter()
            .find(|c| c.codec_id == codec_id)
            .ok_or_else(|| Error::Format(format!("codec_params: unknown codec {}", codec_id.0)))?;
        let bytes = read_segment_payload_typed(
            &mut self.file.lock(),
            codec.params_ref,
            SegmentType::CodecParams,
        )?;
        CodecParams::decode(codec.family, &bytes)
    }

    /// [`Self::codec_params`] narrowed to `QamLloydMax`; errors if the
    /// codec belongs to another family.
    pub fn codec_qam_params(&self, codec_id: CodecId) -> Result<QamLloydMaxParams> {
        match self.codec_params(codec_id)? {
            CodecParams::QamLloydMax(p) => Ok(p),
            other => Err(Error::Format(format!(
                "codec_qam_params: codec {} is {:?}, not QamLloydMax",
                codec_id.0,
                other.family()
            ))),
        }
    }

    /// [`Self::codec_params`] narrowed to `Upq`; errors if the codec
    /// belongs to another family.
    pub fn codec_upq_params(&self, codec_id: CodecId) -> Result<upq_params::UpqParams> {
        match self.codec_params(codec_id)? {
            CodecParams::Upq(p) => Ok(p),
            other => Err(Error::Format(format!(
                "codec_upq_params: codec {} is {:?}, not Upq",
                codec_id.0,
                other.family()
            ))),
        }
    }
}

/// Content-addressed codec fingerprint: BLAKE3 over the family's stable
/// wire discriminant, the params wire version, and the raw params bytes.
pub(super) fn compute_codec_deterministic_id(
    params: &CodecParams,
    params_payload: &[u8],
) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&params.family_discriminant().to_le_bytes());
    hasher.update(&params.wire_version().to_le_bytes());
    hasher.update(params_payload);
    *hasher.finalize().as_bytes()
}

/// Open-time codec cache: decode every catalog codec's params segment
/// and instantiate the live codec. The deterministic id recorded in the
/// catalog is authoritative; instances are content-derived from params.
pub(super) fn build_codec_cache(
    file: &mut File,
    codecs: &[CodecDesc],
) -> Result<HashMap<CodecId, Box<dyn VectorCodec>>> {
    let mut cache: HashMap<CodecId, Box<dyn VectorCodec>> = HashMap::new();
    for codec in codecs {
        codec.validate()?;
        let params_bytes =
            read_segment_payload_typed(file, codec.params_ref, SegmentType::CodecParams)?;
        let params = CodecParams::decode(codec.family, &params_bytes).map_err(|e| {
            Error::Unsupported(format!(
                "build_codec_cache: codec {}: {e}",
                codec.codec_id.0
            ))
        })?;
        cache.insert(codec.codec_id, params.instantiate()?);
    }
    Ok(cache)
}
