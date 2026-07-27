// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! `QamLloydMaxParams` payload codec, spec §14.
//!
//! Magic `NXQL`, version 1. Hand-coded wire format — a Rust struct-field
//! reorder must not silently break compatibility.
//!
//! ## What this codec is
//!
//! QAM Lloyd-Max is a complex-pair vector codec:
//!
//! 1. Apply a fixed orthogonal rotation `H` (block-Hadamard with sign
//!    mask) to the input vector.
//! 2. Pair adjacent rotated coordinates into complex `z_k = a_k e^(i θ_k)`.
//! 3. Quantize the amplitude `a_k / σ_k` with an `N_a`-level Lloyd-Max
//!    scalar quantizer for the unit Rayleigh PDF
//!    (`N_a = 2^amp_bits`).
//! 4. Quantize the phase `θ_k` with a uniform `N_p`-level quantizer
//!    on `(-π, π]` (`N_p = 2^phase_bits`).
//!
//! The persisted parameters fully describe the encoder/decoder:
//! the sign mask is derived deterministically from `rotation_seed`,
//! the Lloyd-Max codebook is a closed-form function of `amp_bits`,
//! and `sigma_per_pair` is the only data-dependent table (one f32 per
//! complex pair, `dimension / 2` floats total).
//!
//! Wire layout (little-endian):
//! ```text
//! [magic NXQL                : 4 B            ]
//! [version u16 = 1           : 2 B            ]
//! [dimension u32             : 4 B            ]
//! [num_pairs u32             : 4 B            ] (= dimension / 2)
//! [block_size u32            : 4 B            ]
//! [rotation_seed u64         : 8 B            ]
//! [amp_bits u8               : 1 B            ]
//! [phase_bits u8             : 1 B            ]
//! [renormalize_at_decode u8  : 1 B            ] (0 = false, 1 = true)
//! [reserved u8 = 0           : 1 B            ]
//! [sigma_count u32           : 4 B            ] (= num_pairs)
//! [sigma_per_pair: f32 × P   : 4·num_pairs B  ]
//! [calibration_id            : 32 B           ]
//! ```

use crate::{
    error::{Error, Result},
    format::Checksum,
};

const MAGIC: [u8; 4] = *b"VLQL";
const VERSION: u16 = 1;

/// Inclusive minimum / maximum bits permitted for amplitude or phase.
/// 1 bit lower bound is theoretical (one level), 8 is the practical cap
/// — a single byte index per pair, also the largest LUT we want to keep
/// resident in the SIMD asymmetric kernel.
pub const QAM_BITS_MIN: u8 = 1;
pub const QAM_BITS_MAX: u8 = 8;

/// Persisted parameters for a `QamLloydMax` codec instance.
#[derive(Clone, Debug, PartialEq)]
pub struct QamLloydMaxParams {
    /// Vector dimension. Must be a positive even number; we pair adjacent
    /// coordinates into complex pairs, so dimension must be 2·num_pairs.
    pub dimension: u32,
    /// Number of complex pairs (= `dimension / 2`). Persisted for
    /// validator clarity even though it is derivable.
    pub num_pairs: u32,
    /// Block size for the block-Hadamard rotation. Must be a power of two
    /// and must divide `dimension`. Default in the reference Python
    /// codec is 1024.
    pub block_size: u32,
    /// Seed driving the deterministic ±1 sign mask consumed by the
    /// block-Hadamard rotation. We use SplitMix64 (spec §14.4.4) to
    /// expand this seed into a per-coordinate sign sequence.
    pub rotation_seed: u64,
    /// Bits per amplitude index. Number of Lloyd-Max levels = `1 << amp_bits`.
    pub amp_bits: u8,
    /// Bits per phase index. Number of uniform phase levels = `1 << phase_bits`.
    pub phase_bits: u8,
    /// If `true`, the decoder divides the reconstructed vector by its
    /// L2 norm before returning. Asymmetric distance kernels use this
    /// to decide whether to divide accumulated dot products by the
    /// stored encode-time norm.
    pub renormalize_at_decode: bool,
    /// One f32 per complex pair: estimated Rayleigh `σ_k` from the
    /// calibration sample. Used to scale amplitudes into the unit
    /// Lloyd-Max codebook at encode time and to scale them back at
    /// decode time.
    pub sigma_per_pair: Vec<f32>,
    /// BLAKE3 of the corpus sample used for calibration. Required for
    /// reproducibility — different samples yield different `sigma_per_pair`.
    pub calibration_id: Checksum,
}

pub(crate) fn encode_qam_lloyd_max_params(params: &QamLloydMaxParams) -> Result<Vec<u8>> {
    validate(params)?;

    let mut buf =
        Vec::with_capacity(4 + 2 + 4 + 4 + 4 + 8 + 4 + 4 + 4 * params.sigma_per_pair.len() + 32);
    buf.extend_from_slice(&MAGIC);
    buf.extend_from_slice(&VERSION.to_le_bytes());
    buf.extend_from_slice(&params.dimension.to_le_bytes());
    buf.extend_from_slice(&params.num_pairs.to_le_bytes());
    buf.extend_from_slice(&params.block_size.to_le_bytes());
    buf.extend_from_slice(&params.rotation_seed.to_le_bytes());
    buf.push(params.amp_bits);
    buf.push(params.phase_bits);
    buf.push(if params.renormalize_at_decode { 1 } else { 0 });
    buf.push(0u8); // reserved
    buf.extend_from_slice(&(params.sigma_per_pair.len() as u32).to_le_bytes());
    for s in &params.sigma_per_pair {
        buf.extend_from_slice(&s.to_le_bytes());
    }
    buf.extend_from_slice(&params.calibration_id);
    Ok(buf)
}

pub(crate) fn decode_qam_lloyd_max_params(bytes: &[u8]) -> Result<QamLloydMaxParams> {
    let mut cur = Cursor::new(bytes);
    let magic = cur.read_array::<4>()?;
    if magic != MAGIC {
        return Err(Error::Format(format!(
            "QamLloydMaxParams magic mismatch: expected NXQL, got {magic:?}"
        )));
    }
    let version = cur.read_u16()?;
    if version != VERSION {
        return Err(Error::Format(format!(
            "QamLloydMaxParams version mismatch: expected {VERSION}, got {version}"
        )));
    }
    let dimension = cur.read_u32()?;
    let num_pairs = cur.read_u32()?;
    let block_size = cur.read_u32()?;
    let rotation_seed = u64::from_le_bytes(cur.read_array::<8>()?);
    let amp_bits = cur.read_u8()?;
    let phase_bits = cur.read_u8()?;
    let renorm_byte = cur.read_u8()?;
    if renorm_byte > 1 {
        return Err(Error::Format(format!(
            "QamLloydMaxParams renormalize_at_decode must be 0 or 1, got {renorm_byte}"
        )));
    }
    let renormalize_at_decode = renorm_byte == 1;
    let reserved = cur.read_u8()?;
    if reserved != 0 {
        return Err(Error::Format(format!(
            "QamLloydMaxParams reserved byte must be 0, got {reserved}"
        )));
    }
    let sigma_count = cur.read_u32()?;
    if sigma_count != num_pairs {
        return Err(Error::Format(format!(
            "QamLloydMaxParams sigma_count ({sigma_count}) must equal num_pairs ({num_pairs})"
        )));
    }
    let mut sigma_per_pair = Vec::with_capacity(sigma_count as usize);
    for _ in 0..sigma_count {
        sigma_per_pair.push(f32::from_le_bytes(cur.read_array::<4>()?));
    }
    let calibration_id = cur.read_array::<32>()?;
    if !cur.is_empty() {
        return Err(Error::Format(format!(
            "QamLloydMaxParams: {} trailing bytes",
            cur.remaining()
        )));
    }

    let params = QamLloydMaxParams {
        dimension,
        num_pairs,
        block_size,
        rotation_seed,
        amp_bits,
        phase_bits,
        renormalize_at_decode,
        sigma_per_pair,
        calibration_id,
    };
    validate(&params)?;
    Ok(params)
}

fn validate(params: &QamLloydMaxParams) -> Result<()> {
    if params.dimension == 0 {
        return Err(Error::Format(
            "QamLloydMaxParams dimension must be > 0".into(),
        ));
    }
    if !params.dimension.is_multiple_of(2) {
        return Err(Error::Format(format!(
            "QamLloydMaxParams dimension ({}) must be even (paired into complex)",
            params.dimension
        )));
    }
    if params.num_pairs != params.dimension / 2 {
        return Err(Error::Format(format!(
            "QamLloydMaxParams num_pairs ({}) must equal dimension/2 ({})",
            params.num_pairs,
            params.dimension / 2
        )));
    }
    if params.block_size == 0 {
        return Err(Error::Format(
            "QamLloydMaxParams block_size must be > 0".into(),
        ));
    }
    if !params.block_size.is_power_of_two() {
        return Err(Error::Format(format!(
            "QamLloydMaxParams block_size ({}) must be a power of two",
            params.block_size
        )));
    }
    if !params.dimension.is_multiple_of(params.block_size) {
        return Err(Error::Format(format!(
            "QamLloydMaxParams block_size ({}) must divide dimension ({})",
            params.block_size, params.dimension
        )));
    }
    if !(QAM_BITS_MIN..=QAM_BITS_MAX).contains(&params.amp_bits) {
        return Err(Error::Format(format!(
            "QamLloydMaxParams amp_bits ({}) must be in [{QAM_BITS_MIN}, {QAM_BITS_MAX}]",
            params.amp_bits
        )));
    }
    if !(QAM_BITS_MIN..=QAM_BITS_MAX).contains(&params.phase_bits) {
        return Err(Error::Format(format!(
            "QamLloydMaxParams phase_bits ({}) must be in [{QAM_BITS_MIN}, {QAM_BITS_MAX}]",
            params.phase_bits
        )));
    }
    if params.sigma_per_pair.len() != params.num_pairs as usize {
        return Err(Error::Format(format!(
            "QamLloydMaxParams sigma_per_pair length ({}) must equal num_pairs ({})",
            params.sigma_per_pair.len(),
            params.num_pairs
        )));
    }
    for (i, s) in params.sigma_per_pair.iter().enumerate() {
        if !s.is_finite() {
            return Err(Error::Format(format!(
                "QamLloydMaxParams sigma_per_pair[{i}] is not finite"
            )));
        }
        if *s <= 0.0 {
            return Err(Error::Format(format!(
                "QamLloydMaxParams sigma_per_pair[{i}] = {s} must be positive"
            )));
        }
    }
    Ok(())
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len() - self.pos
    }

    fn is_empty(&self) -> bool {
        self.pos == self.bytes.len()
    }

    fn read_array<const N: usize>(&mut self) -> Result<[u8; N]> {
        if self.pos + N > self.bytes.len() {
            return Err(Error::Format(format!(
                "QamLloydMaxParams: truncated, need {N} bytes at offset {}",
                self.pos
            )));
        }
        let mut out = [0u8; N];
        out.copy_from_slice(&self.bytes[self.pos..self.pos + N]);
        self.pos += N;
        Ok(out)
    }

    fn read_u8(&mut self) -> Result<u8> {
        Ok(self.read_array::<1>()?[0])
    }

    fn read_u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.read_array::<2>()?))
    }

    fn read_u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.read_array::<4>()?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_params(dim: u32, block_size: u32, amp_bits: u8, phase_bits: u8) -> QamLloydMaxParams {
        let num_pairs = dim / 2;
        QamLloydMaxParams {
            dimension: dim,
            num_pairs,
            block_size,
            rotation_seed: 0x0123_4567_89AB_CDEF,
            amp_bits,
            phase_bits,
            renormalize_at_decode: true,
            sigma_per_pair: vec![0.025_f32; num_pairs as usize],
            calibration_id: [0xAB; 32],
        }
    }

    #[test]
    fn roundtrip() {
        let params = sample_params(3072, 1024, 5, 6);
        let bytes = encode_qam_lloyd_max_params(&params).expect("encode");
        let decoded = decode_qam_lloyd_max_params(&bytes).expect("decode");
        assert_eq!(decoded, params);
    }

    #[test]
    fn deterministic_encode_is_byte_identical() {
        let params = sample_params(3072, 1024, 5, 6);
        let a = encode_qam_lloyd_max_params(&params).expect("encode a");
        let b = encode_qam_lloyd_max_params(&params).expect("encode b");
        assert_eq!(a, b);
    }

    #[test]
    fn renormalize_false_roundtrips() {
        let mut params = sample_params(3072, 1024, 5, 6);
        params.renormalize_at_decode = false;
        let bytes = encode_qam_lloyd_max_params(&params).expect("encode");
        let decoded = decode_qam_lloyd_max_params(&bytes).expect("decode");
        assert!(!decoded.renormalize_at_decode);
    }

    #[test]
    fn rejects_magic_mismatch() {
        let params = sample_params(3072, 1024, 5, 6);
        let mut bytes = encode_qam_lloyd_max_params(&params).expect("encode");
        bytes[0] = b'X';
        let err = decode_qam_lloyd_max_params(&bytes).expect_err("magic mismatch");
        assert!(err.to_string().contains("magic"));
    }

    #[test]
    fn rejects_version_mismatch() {
        let params = sample_params(3072, 1024, 5, 6);
        let mut bytes = encode_qam_lloyd_max_params(&params).expect("encode");
        bytes[4..6].copy_from_slice(&99u16.to_le_bytes());
        let err = decode_qam_lloyd_max_params(&bytes).expect_err("version mismatch");
        assert!(err.to_string().contains("version"));
    }

    #[test]
    fn rejects_zero_dimension() {
        let mut params = sample_params(3072, 1024, 5, 6);
        params.dimension = 0;
        params.num_pairs = 0;
        params.sigma_per_pair.clear();
        let err = encode_qam_lloyd_max_params(&params).expect_err("zero dim");
        assert!(err.to_string().contains("dimension"));
    }

    #[test]
    fn rejects_odd_dimension() {
        let mut params = sample_params(3072, 1024, 5, 6);
        params.dimension = 3071;
        let err = encode_qam_lloyd_max_params(&params).expect_err("odd dim");
        assert!(err.to_string().contains("even"));
    }

    #[test]
    fn rejects_num_pairs_dimension_mismatch() {
        let mut params = sample_params(3072, 1024, 5, 6);
        params.num_pairs = 999;
        let err = encode_qam_lloyd_max_params(&params).expect_err("num_pairs mismatch");
        assert!(err.to_string().contains("num_pairs"));
    }

    #[test]
    fn rejects_non_power_of_two_block_size() {
        let mut params = sample_params(3072, 1024, 5, 6);
        params.block_size = 1000;
        let err = encode_qam_lloyd_max_params(&params).expect_err("block_size");
        assert!(err.to_string().contains("power of two"));
    }

    #[test]
    fn rejects_block_size_not_dividing_dim() {
        let mut params = sample_params(3072, 1024, 5, 6);
        params.block_size = 2048; // doesn't divide 3072
        let err = encode_qam_lloyd_max_params(&params).expect_err("block_size divides");
        assert!(err.to_string().contains("divide"));
    }

    #[test]
    fn rejects_amp_bits_out_of_range() {
        let mut params = sample_params(3072, 1024, 0, 6);
        let err = encode_qam_lloyd_max_params(&params).expect_err("amp_bits low");
        assert!(err.to_string().contains("amp_bits"));
        params.amp_bits = 9;
        let err = encode_qam_lloyd_max_params(&params).expect_err("amp_bits high");
        assert!(err.to_string().contains("amp_bits"));
    }

    #[test]
    fn rejects_phase_bits_out_of_range() {
        let mut params = sample_params(3072, 1024, 5, 0);
        let err = encode_qam_lloyd_max_params(&params).expect_err("phase_bits low");
        assert!(err.to_string().contains("phase_bits"));
        params.phase_bits = 9;
        let err = encode_qam_lloyd_max_params(&params).expect_err("phase_bits high");
        assert!(err.to_string().contains("phase_bits"));
    }

    #[test]
    fn rejects_sigma_length_mismatch() {
        let mut params = sample_params(3072, 1024, 5, 6);
        params.sigma_per_pair.pop();
        let err = encode_qam_lloyd_max_params(&params).expect_err("sigma length");
        assert!(err.to_string().contains("sigma_per_pair"));
    }

    #[test]
    fn rejects_non_finite_sigma() {
        let mut params = sample_params(3072, 1024, 5, 6);
        params.sigma_per_pair[7] = f32::NAN;
        let err = encode_qam_lloyd_max_params(&params).expect_err("nan sigma");
        assert!(err.to_string().contains("finite"));
    }

    #[test]
    fn rejects_nonpositive_sigma() {
        let mut params = sample_params(3072, 1024, 5, 6);
        params.sigma_per_pair[3] = 0.0;
        let err = encode_qam_lloyd_max_params(&params).expect_err("zero sigma");
        assert!(err.to_string().contains("positive"));
    }

    #[test]
    fn rejects_trailing_bytes() {
        let params = sample_params(3072, 1024, 5, 6);
        let mut bytes = encode_qam_lloyd_max_params(&params).expect("encode");
        bytes.push(0xCD);
        let err = decode_qam_lloyd_max_params(&bytes).expect_err("trailing");
        assert!(err.to_string().contains("trailing"));
    }

    #[test]
    fn rejects_invalid_renorm_byte() {
        let params = sample_params(3072, 1024, 5, 6);
        let mut bytes = encode_qam_lloyd_max_params(&params).expect("encode");
        // renormalize_at_decode is at offset 4 + 2 + 4 + 4 + 4 + 8 + 1 + 1 = 28.
        bytes[28] = 5;
        let err = decode_qam_lloyd_max_params(&bytes).expect_err("renorm byte");
        assert!(err.to_string().contains("renormalize"));
    }

    #[test]
    fn small_dim_round_trip() {
        let params = sample_params(64, 32, 4, 6);
        let bytes = encode_qam_lloyd_max_params(&params).expect("encode small");
        let decoded = decode_qam_lloyd_max_params(&bytes).expect("decode small");
        assert_eq!(decoded, params);
    }
}
