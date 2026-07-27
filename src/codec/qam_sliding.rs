//! qam_sliding — register-resident TBL lookups + i8 integer SDOT kernel.
//!
//! Production fast scoring path for QAM Lloyd-Max. Storage on disk is
//! identical to the canonical packed qam codec (5+6 packed, 2112 B/vec
//! at dim=3072 ⇒ 402.8 MiB for 200 k vectors). The kernel below is what
//! makes asymmetric distance fast:
//!
//!   1. **Codebook lives in NEON registers, not L1d.**
//!      `amp_table` (32 i8) → 2 NEON registers (`int8x16x2_t`).
//!      `cos_table` (64 i8) → 4 NEON registers (`int8x16x4_t`).
//!      `sin_table` (64 i8) → 4 NEON registers (`int8x16x4_t`).
//!      Inner-loop lookups are `vqtbl2_s8` / `vqtbl4_s8` — 1-cycle
//!      register-only TBL gathers. Zero L1d traffic for the codebook.
//!
//!   2. **i8-quantized query.** The prepared query is `[re, im, re, im,
//!      …]` of `i8`s with a single f32 dequant scale, sized to fit the
//!      kernel's i8 multiply-accumulate chain.
//!
//!   3. **Integer reduction via SDOT.** The inner loop produces an
//!      `int8x8` vector of `(q_re·c + q_im·s)` values, then SDOTs
//!      against the i8 amplitudes — 1 cycle for 8 i8×i8 multiplications
//!      reduced into 2 i32 lanes.
//!
//!   4. **Elide-norm by default** — same as v2-elide. Ranks by raw
//!      inner product; recall hit ~0.001 vs full-norm.
//!
//! ## A note on the user-described "11-instruction holy grail"
//!
//! The original recipe asked for `vqrdmulh_s8` / `vqrdmlah_s8` to chain
//! three i8 multiplications (`q_re·c → +q_im·s → ·a`) without ever
//! widening. **Those intrinsics don't exist on aarch64**: ARM A64 only
//! defines SQRDMULH and SQRDMLAH for `H` (i16) and `S` (i32) elements.
//! Rust's `core::arch::aarch64` reflects that absence.
//!
//! Practical replacement that *is* on the hardware:
//! ```text
//! ip16  = q_re_i8 * c_i8 + q_im_i8 * s_i8        (vmull_s8 + vmlal_s8)
//! ip8   = sat(ip16 >> 7)                          (vqshrn_n_s16)
//! acc32 += dot4(ip8, a_i8)                        (sdot, inline asm)
//! ```
//! Cost: one extra widen-narrow trip vs the impossible all-i8 chain,
//! but everything else holds.
//!
//! ## Per-iteration count (8 pairs)
//!
//! Scalar work (extracting 8 ai + 8 pi from two `u64`s): 16 shifts +
//! 16 masks. Apple Silicon retires those at ~6 per cycle.
//!
//! NEON kernel: 11 instructions per 8 pairs (≈ 1.4 cycles/pair on
//! 6-wide issue, lower with ILP across multiple iterations in flight):
//!  - 2× `vld1_u8` (indices into NEON)
//!  - 1× `vqtbl2_s8` (amp lookup)
//!  - 2× `vqtbl4_s8` (cos / sin lookups)
//!  - 1× `vld2_s8`   (q_re / q_im deinterleaved)
//!  - 1× `vmull_s8`  (q_re·c → i16x8)
//!  - 1× `vmlal_s8`  (i16x8 += q_im·s)
//!  - 1× `vqshrn_n_s16` (i16x8 → i8x8 via `>>7` saturating-narrow)
//!  - 1× SDOT (8 i8×i8 → 2 i32 lanes)
//!  - 1× pointer advance (folded by the compiler)

mod simd;

use crate::codec::qam_lloyd_max::{QamLloydMaxCodec, block_hadamard_forward};
use crate::error::{Error, Result};
use crate::format::catalog::VectorMetric;

const N_AMP: usize = 32;
const N_PHASE: usize = 64;

/// i8-quantized prepared query: `[re_0, im_0, re_1, im_1, …]` interleaved.
pub struct QamSlidingPreparedQuery {
    /// Length `2 · num_pairs`. Each pair is `(re, im)` adjacent.
    pub q_i8: Vec<i8>,
    /// Precomputed `q_scale * cos_sin_scale * amp_scale * 128.0` — the
    /// single f32 multiplier the kernel applies to `dot_int` to get
    /// real-scale cosine. Computed once at prepare_query time so the
    /// per-candidate inner loop doesn't redo three fmul + 1 const-mul
    /// every iteration. Zoom-style query-independent precompute.
    pub combined_scale: f32,
}

#[derive(Debug)]
pub(crate) struct QamSlidingEngine {
    pub dim: usize,
    pub num_pairs: usize,
    pub block_size: usize,
    pub base_bytes_per_vector: usize,
    pub amp_stream_offset: usize,
    pub amp_stream_bytes: usize,
    pub phase_stream_offset: usize,
    pub phase_stream_bytes: usize,
    sigma_per_pair: Vec<f32>,
    signs: Vec<f32>,
    amp_levels_unit: Vec<f32>,
    /// 32 amp levels quantized to i8 in `[0, 127]`. Loaded as
    /// `int8x16x2_t` (2 NEON registers) at the top of the kernel.
    amp_table_i8: [i8; N_AMP],
    /// 64 cos values quantized to half-i8 range `[-63, 63]`.
    cos_table_i8: [i8; N_PHASE],
    /// 64 sin values quantized to half-i8 range `[-63, 63]`.
    sin_table_i8: [i8; N_PHASE],
    /// `max(amp_levels_unit) / 127`.
    amp_scale: f32,
    /// `1 / 63` (cos / sin range).
    cos_sin_scale: f32,
}

impl QamSlidingEngine {
    pub(crate) fn from_codec(src: &QamLloydMaxCodec) -> Result<Self> {
        if src.amp_bits != 5 || src.phase_bits != 6 {
            return Err(Error::Format(format!(
                "QamSlidingEngine requires amp_bits=5 and phase_bits=6 (got {}, {})",
                src.amp_bits, src.phase_bits
            )));
        }
        debug_assert!(src.num_pairs.is_multiple_of(8));

        let amp_max = src
            .amp_levels_unit
            .iter()
            .copied()
            .fold(0f32, f32::max)
            .max(1e-12);
        let amp_scale = amp_max / 127.0;
        let amp_inv = 1.0 / amp_scale;
        let mut amp_table_i8 = [0i8; N_AMP];
        for (dst, &level) in amp_table_i8.iter_mut().zip(src.amp_levels_unit.iter()) {
            *dst = (level * amp_inv).round().clamp(0.0, 127.0) as i8;
        }
        let mut cos_table_i8 = [0i8; N_PHASE];
        let mut sin_table_i8 = [0i8; N_PHASE];
        for i in 0..N_PHASE {
            cos_table_i8[i] = (src.phase_cos_lut[i] * 63.0).round().clamp(-63.0, 63.0) as i8;
            sin_table_i8[i] = (src.phase_sin_lut[i] * 63.0).round().clamp(-63.0, 63.0) as i8;
        }

        Ok(Self {
            dim: src.dim,
            num_pairs: src.num_pairs,
            block_size: src.block_size,
            base_bytes_per_vector: src.base_bytes_per_vector,
            amp_stream_offset: src.amp_stream_offset,
            amp_stream_bytes: src.amp_stream_bytes,
            phase_stream_offset: src.phase_stream_offset,
            phase_stream_bytes: src.phase_stream_bytes,
            sigma_per_pair: src.sigma_per_pair.clone(),
            signs: src.signs.clone(),
            amp_levels_unit: src.amp_levels_unit.clone(),
            amp_table_i8,
            cos_table_i8,
            sin_table_i8,
            amp_scale,
            cos_sin_scale: 1.0 / 63.0,
        })
    }

    pub fn prepare_query(&self, query: &[f32]) -> Result<QamSlidingPreparedQuery> {
        if query.len() != self.dim {
            return Err(Error::Format(format!(
                "QamSlidingEngine::prepare_query: dim mismatch ({} vs {})",
                query.len(),
                self.dim
            )));
        }
        for &v in query {
            if !v.is_finite() {
                return Err(Error::Format(
                    "QamSlidingEngine::prepare_query: non-finite input".into(),
                ));
            }
        }
        let mut q_rot = query.to_vec();
        block_hadamard_forward(&mut q_rot, &self.signs, self.block_size)?;

        // σ-fuse and find max abs in one pass.
        let mut q_re = vec![0f32; self.num_pairs];
        let mut q_im = vec![0f32; self.num_pairs];
        let mut max_abs = 0f32;
        for k in 0..self.num_pairs {
            let s = self.sigma_per_pair[k];
            let re = q_rot[2 * k] * s;
            let im = q_rot[2 * k + 1] * s;
            q_re[k] = re;
            q_im[k] = im;
            max_abs = max_abs.max(re.abs()).max(im.abs());
        }
        let q_scale = (max_abs / 127.0).max(1e-30);
        let q_inv = 1.0 / q_scale;
        let mut q_i8 = Vec::with_capacity(2 * self.num_pairs);
        for k in 0..self.num_pairs {
            q_i8.push((q_re[k] * q_inv).round().clamp(-127.0, 127.0) as i8);
            q_i8.push((q_im[k] * q_inv).round().clamp(-127.0, 127.0) as i8);
        }
        let combined_scale = q_scale * self.cos_sin_scale * self.amp_scale * 128.0;
        Ok(QamSlidingPreparedQuery {
            q_i8,
            combined_scale,
        })
    }

    /// Returns the integer dot-product accumulator (apply the dequant
    /// scale at the caller). This is a pure inner-product; no `‖ŷ‖²`
    /// is computed (elide-norm is baked in).
    #[inline]
    pub fn raw_dot_int(&self, prep: &QamSlidingPreparedQuery, base: &[u8]) -> i64 {
        debug_assert_eq!(base.len(), self.base_bytes_per_vector);
        let amp_stream =
            &base[self.amp_stream_offset..self.amp_stream_offset + self.amp_stream_bytes];
        let phase_stream =
            &base[self.phase_stream_offset..self.phase_stream_offset + self.phase_stream_bytes];

        #[cfg(target_arch = "aarch64")]
        unsafe {
            return simd::aarch64::raw_dot_int(
                &self.amp_table_i8,
                &self.cos_table_i8,
                &self.sin_table_i8,
                self.num_pairs,
                amp_stream,
                phase_stream,
                &prep.q_i8,
            );
        }

        // The AVX2 kernel is bit-exactly equivalent to
        // `raw_dot_int_scalar` — both reproduce NEON's
        // `vqshrn_n_s16(x,7)` rounding-saturating narrow. See §4.3 +
        // §11: cross-arch replay determinism is a correctness
        // invariant, not just a performance choice.
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        unsafe {
            // Static dispatch — target_feature("avx2") guaranteed at
            // compile time, no runtime check needed.
            return simd::x86_64::raw_dot_int(
                &self.amp_table_i8,
                &self.cos_table_i8,
                &self.sin_table_i8,
                self.num_pairs,
                amp_stream,
                phase_stream,
                &prep.q_i8,
            );
        }
        #[cfg(all(target_arch = "x86_64", not(target_feature = "avx2")))]
        {
            if std::is_x86_feature_detected!("avx2") {
                unsafe {
                    return simd::x86_64::raw_dot_int(
                        &self.amp_table_i8,
                        &self.cos_table_i8,
                        &self.sin_table_i8,
                        self.num_pairs,
                        amp_stream,
                        phase_stream,
                        &prep.q_i8,
                    );
                }
            }
        }

        #[allow(unreachable_code)]
        {
            self.raw_dot_int_scalar(amp_stream, phase_stream, &prep.q_i8)
        }
    }

    #[cfg_attr(any(test, feature = "bench"), allow(dead_code))]
    pub(crate) fn raw_dot_int_scalar(
        &self,
        amp_stream: &[u8],
        phase_stream: &[u8],
        q_i8: &[i8],
    ) -> i64 {
        // Reference impl that mirrors the NEON kernel's quantization
        // (including the `>>7` narrowing, so results are bit-comparable).
        let mut acc: i64 = 0;
        let groups = self.num_pairs / 8;
        for g in 0..groups {
            let amp_off = g * 5;
            let phase_off = g * 6;
            let mut a_tail = [0u8; 8];
            let mut p_tail = [0u8; 8];
            let amp_avail = (amp_stream.len() - amp_off).min(8);
            let phase_avail = (phase_stream.len() - phase_off).min(8);
            a_tail[..amp_avail].copy_from_slice(&amp_stream[amp_off..amp_off + amp_avail]);
            p_tail[..phase_avail]
                .copy_from_slice(&phase_stream[phase_off..phase_off + phase_avail]);
            let a_chunk = u64::from_le_bytes(a_tail);
            let p_chunk = u64::from_le_bytes(p_tail);
            let q_base = g * 16;
            for i in 0..8 {
                let ai = ((a_chunk >> (i * 5)) & 0x1F) as usize;
                let pi = ((p_chunk >> (i * 6)) & 0x3F) as usize;
                let a = self.amp_table_i8[ai] as i32;
                let c = self.cos_table_i8[pi] as i32;
                let s = self.sin_table_i8[pi] as i32;
                let qr = q_i8[q_base + i * 2] as i32;
                let qi = q_i8[q_base + i * 2 + 1] as i32;
                let ip16 = qr * c + qi * s;
                let ip8 = ((ip16 + (1 << 6)) >> 7).clamp(-128, 127);
                acc += (ip8 * a) as i64;
            }
        }
        acc
    }

    pub fn asymmetric_distance(
        &self,
        prep: &QamSlidingPreparedQuery,
        base: &[u8],
        _metric: VectorMetric,
    ) -> Result<f32> {
        if base.len() != self.base_bytes_per_vector {
            return Err(Error::Format(format!(
                "QamSlidingEngine::asymmetric_distance: base length mismatch ({} vs {})",
                base.len(),
                self.base_bytes_per_vector
            )));
        }
        // Cosine = -dot / (‖q‖·‖ŷ‖). q_norm is per-query (constant across
        // rows). ŷ_norm is PER ROW and MUST NOT be omitted — see the
        // valise-sliding-bug-repro bench. On the slow path we recompute it
        // from the amp_stream; production callers should use
        // `asymmetric_distance_with_norm` with a pre-cached table.
        let y_hat_norm = self.y_hat_norm(base);
        self.asymmetric_distance_with_norm(prep, base, y_hat_norm)
    }

    /// Hot-path variant for callers that have cached `‖ŷ‖` per row
    /// (e.g. precomputed at corpus load time). Skips the scalar
    /// post-pass that walks `amp_stream` to recompute the norm.
    ///
    /// At dim=768 num_pairs=384 this saves ~384 multiplies + 1 sqrt
    /// per scored row — ~2-3× wall-clock on the brute-force scan.
    pub fn asymmetric_distance_with_norm(
        &self,
        prep: &QamSlidingPreparedQuery,
        base: &[u8],
        y_hat_norm: f32,
    ) -> Result<f32> {
        let inv = if y_hat_norm > 1e-12 {
            1.0 / y_hat_norm
        } else {
            0.0
        };
        self.asymmetric_distance_with_inv_norm(prep, base, inv)
    }

    /// Same as [`asymmetric_distance_with_norm`] but takes
    /// `inv_y_hat_norm = 1.0 / ‖ŷ‖` directly. Production rerank caches
    /// the inverse at file open (one fdiv per *vector*, not per
    /// *candidate-per-query*); saves ~10 cycles × channel_k per query.
    /// Combined-scale is read from `prep` so the per-call body becomes
    /// `dot_int → fmul → fmul`.
    #[inline]
    pub fn asymmetric_distance_with_inv_norm(
        &self,
        prep: &QamSlidingPreparedQuery,
        base: &[u8],
        inv_y_hat_norm: f32,
    ) -> Result<f32> {
        if base.len() != self.base_bytes_per_vector {
            return Err(Error::Format(format!(
                "QamSlidingEngine::asymmetric_distance_with_inv_norm: base length mismatch ({} vs {})",
                base.len(),
                self.base_bytes_per_vector
            )));
        }
        let dot_int = self.raw_dot_int(prep, base);
        let dot_real = dot_int as f32 * prep.combined_scale;
        Ok(-dot_real * inv_y_hat_norm)
    }
    /// Compute `‖ŷ‖` for a single base row directly from its packed amp
    /// indices and the per-pair σ — no f32 decode, no allocation. Matches
    /// the f32 trait path's `y_hat_norm_sq = Σ (σ_k · amp_unit[ai_k])²`.
    /// Public so callers can build a precomputed norm cache (4 B/vec)
    /// once at file open and pass into `asymmetric_distance_with_norm`.
    #[inline]
    pub fn y_hat_norm(&self, base: &[u8]) -> f32 {
        let amp_stream =
            &base[self.amp_stream_offset..self.amp_stream_offset + self.amp_stream_bytes];
        let mut sum_sq: f32 = 0.0;
        let groups = self.num_pairs / 8;
        for g in 0..groups {
            let amp_off = g * 5;
            let mut a_tail = [0u8; 8];
            let avail = (amp_stream.len() - amp_off).min(8);
            a_tail[..avail].copy_from_slice(&amp_stream[amp_off..amp_off + avail]);
            let a_chunk = u64::from_le_bytes(a_tail);
            for i in 0..8 {
                let ai = ((a_chunk >> (i * 5)) & 0x1F) as usize;
                let sigma = self.sigma_per_pair[g * 8 + i];
                let amp_unit = self.amp_levels_unit[ai];
                let v = sigma * amp_unit;
                sum_sq += v * v;
            }
        }
        sum_sq.sqrt()
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
#[allow(clippy::print_stderr, reason = "tests report skipped ISA cases")]
mod x86_64_parity_tests {
    //! Bit-exact AVX2 vs scalar parity for the SDOT kernel
    //! (`docs/X86_64_SIMD_PLAN.md` §8.1 integer rule + §11 cross-
    //! arch replay determinism). Zero tolerance.
    //!
    //! Why bit-exact: `raw_dot_int` feeds the rerank comparator, so
    //! a single-LSB drift can flip near-tied candidate order and
    //! change top-k recall. Top-k orderings must be byte-identical
    //! across arches for the same index + query.
    use super::*;
    use crate::codec::prng::SplitMix64;
    use crate::codec::qam_lloyd_max::QamLloydMaxCodec;

    fn avx2_available() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut prng = SplitMix64::new(seed);
        let mut v = vec![0.0f32; dim];
        let mut i = 0;
        while i + 1 < dim {
            let u1 = ((prng.next_u64() >> 11) as f64 / (1u64 << 53) as f64).max(1e-300);
            let u2 = (prng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            v[i] = (r * theta.cos()) as f32;
            v[i + 1] = (r * theta.sin()) as f32;
            i += 2;
        }
        let n2: f32 = v.iter().map(|&x| x * x).sum();
        let inv = 1.0 / n2.sqrt().max(1e-30);
        for x in v.iter_mut() {
            *x *= inv;
        }
        v
    }

    /// Build a calibrated `(codec, engine, sample)` triple sharing
    /// the same calibration. Tests need the codec to encode base
    /// bytes and the engine to prepare queries / call the kernels.
    fn build_codec_engine(
        num_pairs: usize,
        seed: u64,
    ) -> (QamLloydMaxCodec, QamSlidingEngine, Vec<Vec<f32>>) {
        let dim = 2 * num_pairs;
        let mut block_size = 1024;
        while !dim.is_multiple_of(block_size) && block_size > 1 {
            block_size /= 2;
        }
        let mut codec = QamLloydMaxCodec::with_config(dim, block_size, 5, 6, true).unwrap();
        let sample: Vec<Vec<f32>> = (0..64)
            .map(|i| random_unit_vector(dim, seed ^ (i as u64)))
            .collect();
        codec.calibrate(&sample).unwrap();
        let engine = QamSlidingEngine::from_codec(&codec).unwrap();
        (codec, engine, sample)
    }

    fn run_parity_case(num_pairs: usize, seed: u64, label: &str) {
        if !avx2_available() {
            eprintln!("skip: cpu lacks avx2 ({label})");
            return;
        }
        let (codec, engine, sample) = build_codec_engine(num_pairs, seed);
        let base = codec.encode(&sample[0]).unwrap();
        let prep = engine.prepare_query(&sample[1]).unwrap();

        let amp_stream =
            &base[engine.amp_stream_offset..engine.amp_stream_offset + engine.amp_stream_bytes];
        let phase_stream = &base
            [engine.phase_stream_offset..engine.phase_stream_offset + engine.phase_stream_bytes];

        let scalar = engine.raw_dot_int_scalar(amp_stream, phase_stream, &prep.q_i8);
        let avx2 = unsafe {
            super::simd::x86_64::raw_dot_int(
                &engine.amp_table_i8,
                &engine.cos_table_i8,
                &engine.sin_table_i8,
                engine.num_pairs,
                amp_stream,
                phase_stream,
                &prep.q_i8,
            )
        };
        assert_eq!(
            avx2, scalar,
            "{label}: AVX2={avx2} vs scalar={scalar} (bit-exact integer parity required)"
        );
    }

    #[test]
    fn parity_num_pairs_8() {
        run_parity_case(8, 0xA1, "num_pairs=8 (smallest, tail-only path)");
    }

    #[test]
    fn parity_num_pairs_16() {
        run_parity_case(16, 0xA2, "num_pairs=16 (1 main + tail)");
    }

    #[test]
    fn parity_num_pairs_64() {
        run_parity_case(64, 0xA3, "num_pairs=64");
    }

    #[test]
    fn parity_num_pairs_384() {
        run_parity_case(384, 0xA4, "num_pairs=384 (production d=768)");
    }

    #[test]
    fn parity_many_queries_random() {
        // Property-style: 32 different (base, query) pairs at the
        // production size — catches accumulator drift that only
        // manifests at certain input distributions.
        if !avx2_available() {
            return;
        }
        let (codec, engine, sample) = build_codec_engine(384, 0xB1);
        for i in 0..32 {
            let base = codec.encode(&sample[i % sample.len()]).unwrap();
            let prep = engine
                .prepare_query(&sample[(i + 7) % sample.len()])
                .unwrap();
            let amp_stream =
                &base[engine.amp_stream_offset..engine.amp_stream_offset + engine.amp_stream_bytes];
            let phase_stream = &base[engine.phase_stream_offset
                ..engine.phase_stream_offset + engine.phase_stream_bytes];
            let scalar = engine.raw_dot_int_scalar(amp_stream, phase_stream, &prep.q_i8);
            let avx2 = unsafe {
                super::simd::x86_64::raw_dot_int(
                    &engine.amp_table_i8,
                    &engine.cos_table_i8,
                    &engine.sin_table_i8,
                    engine.num_pairs,
                    amp_stream,
                    phase_stream,
                    &prep.q_i8,
                )
            };
            assert_eq!(
                avx2, scalar,
                "drift at iter {i}: avx2={avx2} scalar={scalar}"
            );
        }
    }
}
