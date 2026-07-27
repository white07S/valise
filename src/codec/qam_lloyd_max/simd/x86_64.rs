//! AVX2 + FMA kernels for the QAM Lloyd-Max codec
//! (`docs/X86_64_SIMD_PLAN.md` §4).
//!
//! Currently shipped: [`asymmetric_dot_pairs`] (Phase 3, §4.1). The
//! other kernels are still stubs; Phases 5 and 6 fill them in.
//!
//! Per §4.1 the AVX2 mapping of the NEON kernel is **port-5-bound by
//! the (re, im) deinterleave shuffles**, not FMA-bound — realistic
//! win over scalar is ~2.5×, not the ~3× a naïve FMA-count would
//! suggest. The kernel still produces (dot, norm_sq) up to fp
//! associativity drift (§11 tolerance: 2e-4 relative for the wider
//! 16-way reduction).

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]

use core::arch::x86_64::*;

/// 8-wide horizontal sum: `__m256` of 8 f32 lanes → scalar f32.
/// `hi += lo` once, then `hadd` twice on the resulting `__m128`.
#[target_feature(enable = "avx,sse3")]
#[inline]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
unsafe fn hsum_avx(v: __m256) -> f32 {
    let lo = _mm256_castps256_ps128(v);
    let hi = _mm256_extractf128_ps(v, 1);
    let s = _mm_add_ps(lo, hi);
    let s2 = _mm_hadd_ps(s, s);
    let s3 = _mm_hadd_ps(s2, s2);
    _mm_cvtss_f32(s3)
}

/// 4-wide horizontal sum on `__m128`.
#[target_feature(enable = "sse3")]
#[inline]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
unsafe fn hsum_sse(v: __m128) -> f32 {
    let s = _mm_hadd_ps(v, v);
    let s2 = _mm_hadd_ps(s, s);
    _mm_cvtss_f32(s2)
}

#[target_feature(enable = "avx2,fma")]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
pub(crate) unsafe fn asymmetric_dot_pairs(
    q_rot: &[f32],
    amp_indices: &[u32],
    phase_indices: &[u32],
    sigma_per_pair: &[f32],
    amp_levels_unit: &[f32],
    phase_cos_lut: &[f32],
    phase_sin_lut: &[f32],
) -> (f32, f32) {
    let p = amp_indices.len();
    let q_ptr = q_rot.as_ptr();
    let mut k = 0usize;

    // Stack scratch buffers for the emulated gather. Sized to the
    // 16-pair main loop iteration (§4.1 — the NEON code uses 8 but
    // the AVX2 path is 16-wide for the doubled vector width). We do
    // NOT use `_mm256_i32gather_ps`: on Haswell/Skylake-class chips
    // (and the lab i7-8550U is Kaby Lake R, same family), `vpgatherdps`
    // is microcoded at ~1 lane/cycle — slower than these scalar
    // loads. Re-evaluate only when indices arrive vector-register-
    // resident from a prior compute step.
    let mut amp_buf = [0.0f32; 16];
    let mut cos_buf = [0.0f32; 16];
    let mut sin_buf = [0.0f32; 16];

    // Two parallel 8-wide accumulators mirror the NEON ILP shape:
    // independent FMA chains avoid stalling on Skylake-family's
    // ~4-cycle FMA latency.
    let mut dot_lo = _mm256_setzero_ps();
    let mut dot_hi = _mm256_setzero_ps();
    let mut norm_lo = _mm256_setzero_ps();
    let mut norm_hi = _mm256_setzero_ps();

    // ---- Main loop: 16 pairs/iter ----
    while k + 16 <= p {
        for j in 0..16 {
            let ai = *amp_indices.get_unchecked(k + j) as usize;
            let pi = *phase_indices.get_unchecked(k + j) as usize;
            let sigma = *sigma_per_pair.get_unchecked(k + j);
            amp_buf[j] = sigma * *amp_levels_unit.get_unchecked(ai);
            cos_buf[j] = *phase_cos_lut.get_unchecked(pi);
            sin_buf[j] = *phase_sin_lut.get_unchecked(pi);
        }

        let amp_lo_v = _mm256_loadu_ps(amp_buf.as_ptr());
        let amp_hi_v = _mm256_loadu_ps(amp_buf.as_ptr().add(8));
        let cos_lo_v = _mm256_loadu_ps(cos_buf.as_ptr());
        let cos_hi_v = _mm256_loadu_ps(cos_buf.as_ptr().add(8));
        let sin_lo_v = _mm256_loadu_ps(sin_buf.as_ptr());
        let sin_hi_v = _mm256_loadu_ps(sin_buf.as_ptr().add(8));

        // (re, im) deinterleave for 16 pairs = 32 f32s from q_rot.
        // No AVX2 equivalent of NEON's vld2q_f32 — we pay 4 shuffles
        // + 4 lane-fix permutes per 16 pairs (§11 "no native vld2q
        // on AVX2"). This is the port-5 bottleneck §4.1 warns about.
        let r0 = _mm256_loadu_ps(q_ptr.add(2 * k));
        let r1 = _mm256_loadu_ps(q_ptr.add(2 * k + 8));
        let r2 = _mm256_loadu_ps(q_ptr.add(2 * k + 16));
        let r3 = _mm256_loadu_ps(q_ptr.add(2 * k + 24));
        let (q_re_lo, q_im_lo) = deinterleave_pairs(r0, r1);
        let (q_re_hi, q_im_hi) = deinterleave_pairs(r2, r3);

        //   tmp = q_re·cos + q_im·sin   (mul + fmadd)
        //   dot += tmp·amp              (fmadd)
        //   norm += amp·amp             (fmadd)
        // Two parallel chains (lo/hi) for ILP.
        let tmp_lo = _mm256_fmadd_ps(q_im_lo, sin_lo_v, _mm256_mul_ps(q_re_lo, cos_lo_v));
        let tmp_hi = _mm256_fmadd_ps(q_im_hi, sin_hi_v, _mm256_mul_ps(q_re_hi, cos_hi_v));
        dot_lo = _mm256_fmadd_ps(tmp_lo, amp_lo_v, dot_lo);
        dot_hi = _mm256_fmadd_ps(tmp_hi, amp_hi_v, dot_hi);
        norm_lo = _mm256_fmadd_ps(amp_lo_v, amp_lo_v, norm_lo);
        norm_hi = _mm256_fmadd_ps(amp_hi_v, amp_hi_v, norm_hi);

        k += 16;
    }

    // ---- 8-pair cleanup (single 8-wide accumulator) ----
    let mut dot_acc = _mm256_add_ps(dot_lo, dot_hi);
    let mut norm_acc = _mm256_add_ps(norm_lo, norm_hi);
    if k + 8 <= p {
        for j in 0..8 {
            let ai = *amp_indices.get_unchecked(k + j) as usize;
            let pi = *phase_indices.get_unchecked(k + j) as usize;
            let sigma = *sigma_per_pair.get_unchecked(k + j);
            amp_buf[j] = sigma * *amp_levels_unit.get_unchecked(ai);
            cos_buf[j] = *phase_cos_lut.get_unchecked(pi);
            sin_buf[j] = *phase_sin_lut.get_unchecked(pi);
        }
        let amp = _mm256_loadu_ps(amp_buf.as_ptr());
        let cos = _mm256_loadu_ps(cos_buf.as_ptr());
        let sin = _mm256_loadu_ps(sin_buf.as_ptr());
        let r0 = _mm256_loadu_ps(q_ptr.add(2 * k));
        let r1 = _mm256_loadu_ps(q_ptr.add(2 * k + 8));
        let (q_re, q_im) = deinterleave_pairs(r0, r1);
        let tmp = _mm256_fmadd_ps(q_im, sin, _mm256_mul_ps(q_re, cos));
        dot_acc = _mm256_fmadd_ps(tmp, amp, dot_acc);
        norm_acc = _mm256_fmadd_ps(amp, amp, norm_acc);
        k += 8;
    }

    // Reduce 8-wide accumulators down to scalar before the 4-wide /
    // scalar tails; the tails are tiny and not worth a separate
    // __m128 accumulator path.
    let mut dot = hsum_avx(dot_acc);
    let mut norm_sq = hsum_avx(norm_acc);

    // ---- 4-pair cleanup (__m128 / SSE FMA) ----
    if k + 4 <= p {
        let mut a4 = [0.0f32; 4];
        let mut c4 = [0.0f32; 4];
        let mut s4 = [0.0f32; 4];
        for j in 0..4 {
            let ai = *amp_indices.get_unchecked(k + j) as usize;
            let pi = *phase_indices.get_unchecked(k + j) as usize;
            let sigma = *sigma_per_pair.get_unchecked(k + j);
            a4[j] = sigma * *amp_levels_unit.get_unchecked(ai);
            c4[j] = *phase_cos_lut.get_unchecked(pi);
            s4[j] = *phase_sin_lut.get_unchecked(pi);
        }
        let amp = _mm_loadu_ps(a4.as_ptr());
        let cos = _mm_loadu_ps(c4.as_ptr());
        let sin = _mm_loadu_ps(s4.as_ptr());
        // Load 8 floats and deinterleave into (re, im) 4-lane vectors
        // via two _mm_shuffle_ps with mask 0x88 / 0xDD (no cross-lane
        // permute needed — SSE shuffle is already lane-local on 128b).
        let r0 = _mm_loadu_ps(q_ptr.add(2 * k));
        let r1 = _mm_loadu_ps(q_ptr.add(2 * k + 4));
        let q_re = _mm_shuffle_ps(r0, r1, 0x88);
        let q_im = _mm_shuffle_ps(r0, r1, 0xDD);
        let tmp = _mm_fmadd_ps(q_im, sin, _mm_mul_ps(q_re, cos));
        let dot4 = _mm_fmadd_ps(tmp, amp, _mm_setzero_ps());
        let norm4 = _mm_mul_ps(amp, amp);
        dot += hsum_sse(dot4);
        norm_sq += hsum_sse(norm4);
        k += 4;
    }

    // ---- Scalar tail (0..3 pairs) ----
    while k < p {
        let ai = amp_indices[k] as usize;
        let pi = phase_indices[k] as usize;
        let amp = sigma_per_pair[k] * amp_levels_unit[ai];
        let c = phase_cos_lut[pi];
        let s = phase_sin_lut[pi];
        let q_re = q_rot[2 * k];
        let q_im = q_rot[2 * k + 1];
        dot += q_re * amp * c + q_im * amp * s;
        norm_sq += amp * amp;
        k += 1;
    }

    (dot, norm_sq)
}

/// Deinterleave two consecutive `__m256` of (re, im) pairs into
/// separate `__m256` of re's and im's, both in linear order.
///
/// Input:
///   r0 = [re0, im0, re1, im1, re2, im2, re3, im3]
///   r1 = [re4, im4, re5, im5, re6, im6, re7, im7]
/// Output:
///   re  = [re0, re1, re2, re3, re4, re5, re6, re7]
///   im  = [im0, im1, im2, im3, im4, im5, im6, im7]
///
/// Cost: 2× `_mm256_shuffle_ps` (per-128-bit-lane) + 2×
/// `_mm256_permute4x64_pd` (lane-crossing). All four go through
/// port 5 on Skylake-family — this is the §4.1 port-5 bottleneck.
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
unsafe fn deinterleave_pairs(r0: __m256, r1: __m256) -> (__m256, __m256) {
    // Per-128-lane shuffle pulling lanes 0,2 (re) and 1,3 (im).
    //   re_un (low 128):  [re0, re1, re4, re5]   from r0[0,2]+r1[0,2]
    //   re_un (hi  128):  [re2, re3, re6, re7]   from r0[4,6]+r1[4,6]
    let re_un = _mm256_shuffle_ps(r0, r1, 0x88);
    let im_un = _mm256_shuffle_ps(r0, r1, 0xDD);
    // re_un now holds [A, B, C, D] in 4× u64 lanes where A=(re0,re1),
    // B=(re4,re5), C=(re2,re3), D=(re6,re7). Swap B and C with the
    // (0, 2, 1, 3) 4x64 permute (mask 0xD8).
    let re = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(re_un), 0xD8));
    let im = _mm256_castpd_ps(_mm256_permute4x64_pd(_mm256_castps_pd(im_un), 0xD8));
    (re, im)
}

/// FWHT stages h = 2 .. n/2, operating in place on `block[0..n]`.
/// Shared by the fused forward and inverse kernels (Phase 5, §4.4) —
/// only the h=1 stage and the scale/sign passes differ between them.
///
/// Stage layout:
/// - **h=2** (8-wide, per-128-bit-lane): `_mm256_permute_ps<0x4E>` swaps
///   the two 64-bit halves inside each 128-bit lane, then a sign-mask
///   + add does the butterfly. No cross-lane shuffles.
/// - **h=4** (8-wide, cross-lane): `_mm256_permute2f128_ps<0x01>`
///   swaps the high and low 128-bit halves of the 256-bit register;
///   sign-mask + add does the butterfly. One 256-bit op per 8-element
///   group.
/// - **h ≥ 8**: paired `_mm256_loadu_ps((p, p+h))` + add/sub + store.
///   Direct port of the NEON loop shape.
#[target_feature(enable = "avx2,fma")]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
unsafe fn fwht_stages_h2_plus(bp: *mut f32, n: usize) {
    debug_assert!(n >= 8);

    // Stage h=2: butterfly within each 128-bit lane (groups of 4
    // floats). Per lane:
    //   v = [α, β, γ, δ]  → output [α+γ, β+δ, α-γ, β-δ]
    // swap via `_mm256_permute_ps<0x4E>` gives `[γ, δ, α, β]` per
    // lane; sign-mask `[1, 1, -1, -1]` lets `swap + (v * mask)` land
    // the right combo per lane.
    let sign_h2 = _mm256_setr_ps(1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(bp.add(i));
        let swap = _mm256_permute_ps::<0x4E>(v);
        let v_signed = _mm256_mul_ps(v, sign_h2);
        _mm256_storeu_ps(bp.add(i), _mm256_add_ps(v_signed, swap));
        i += 8;
    }

    // Stage h=4: cross-128-bit-lane butterfly over each 8-element
    // group. Single 256-bit op per group.
    //   v    = [a0..a3, b0..b3]
    //   swap = permute2f128(v, v, 0x01) = [b0..b3, a0..a3]
    //   sign = [+1×4, -1×4]
    //   v·sign + swap = [a0+b0, .., a0-b0, ..]
    let sign_h4 = _mm256_setr_ps(1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0);
    let mut base = 0usize;
    while base + 8 <= n {
        let v = _mm256_loadu_ps(bp.add(base));
        let swap = _mm256_permute2f128_ps::<0x01>(v, v);
        let v_signed = _mm256_mul_ps(v, sign_h4);
        _mm256_storeu_ps(bp.add(base), _mm256_add_ps(v_signed, swap));
        base += 8;
    }

    // Stages h ≥ 8: paired 256-bit loads. j stride = 8 floats; base
    // stride = 2h. Direct port of NEON's h≥4 shape, widened.
    let mut h = 8usize;
    while h < n {
        let mut base = 0usize;
        while base < n {
            let mut j = 0usize;
            while j < h {
                let a = _mm256_loadu_ps(bp.add(base + j));
                let b = _mm256_loadu_ps(bp.add(base + j + h));
                _mm256_storeu_ps(bp.add(base + j), _mm256_add_ps(a, b));
                _mm256_storeu_ps(bp.add(base + j + h), _mm256_sub_ps(a, b));
                j += 8;
            }
            base += 2 * h;
        }
        h *= 2;
    }
}

/// Fused forward block-Hadamard. Stage h=1 folds the random sign
/// multiply into the butterfly, eliminating the separate full-array
/// sign pass. `block.len() == signs.len() == n` and `n.is_power_of_two()`
/// and `n >= 8` (dispatcher guard — `block_size < 8` falls to scalar
/// per §4.4).
#[target_feature(enable = "avx2,fma")]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
pub(super) unsafe fn fwht_forward_fused(block: &mut [f32], signs: &[f32]) {
    let n = block.len();
    debug_assert!(n.is_power_of_two() && n >= 8);
    debug_assert_eq!(signs.len(), n);
    let bp = block.as_mut_ptr();
    let sp = signs.as_ptr();

    // Stage h=1 with sign multiply folded in.
    //   v = raw · sign
    //   swap = within each 64-bit half, swap the two lanes
    //   v_signed = v · [1, -1, 1, -1, ...]
    //   result = v_signed + swap = [a+b, a-b, c+d, c-d, ...]
    // §4.4 trap: do NOT use `_mm256_addsub_ps` — it does `a-b` in
    // even lanes (opposite of what FWHT wants), and the operand-
    // swap to fix it un-saves the cycle. The explicit shuffle +
    // sign-mul + add pattern is bug-free.
    let lane_signs = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let raw = _mm256_loadu_ps(bp.add(i));
        let s = _mm256_loadu_ps(sp.add(i));
        let v = _mm256_mul_ps(raw, s);
        let swap = _mm256_shuffle_ps::<0xB1>(v, v);
        let v_signed = _mm256_mul_ps(v, lane_signs);
        _mm256_storeu_ps(bp.add(i), _mm256_add_ps(v_signed, swap));
        i += 8;
    }

    fwht_stages_h2_plus(bp, n);

    // Final 1/√n scaling pass.
    let scale = (n as f32).sqrt().recip();
    let scale_v = _mm256_set1_ps(scale);
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(bp.add(i));
        _mm256_storeu_ps(bp.add(i), _mm256_mul_ps(v, scale_v));
        i += 8;
    }
}

/// Fused inverse block-Hadamard. Stage h=1 is plain (no sign fold —
/// for the inverse, signs are applied AFTER the butterflies). The
/// final pass fuses `1/√n` scaling and the per-element sign multiply
/// into one load·mul·mul·store.
#[target_feature(enable = "avx2,fma")]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
pub(super) unsafe fn fwht_inverse_fused(block: &mut [f32], signs: &[f32]) {
    let n = block.len();
    debug_assert!(n.is_power_of_two() && n >= 8);
    debug_assert_eq!(signs.len(), n);
    let bp = block.as_mut_ptr();
    let sp = signs.as_ptr();

    // Stage h=1 (plain butterfly, no sign).
    let lane_signs = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(bp.add(i));
        let swap = _mm256_shuffle_ps::<0xB1>(v, v);
        let v_signed = _mm256_mul_ps(v, lane_signs);
        _mm256_storeu_ps(bp.add(i), _mm256_add_ps(v_signed, swap));
        i += 8;
    }

    fwht_stages_h2_plus(bp, n);

    // Final pass: data[i] := data[i] · (1/√n) · signs[i] in one go.
    let scale = (n as f32).sqrt().recip();
    let scale_v = _mm256_set1_ps(scale);
    let mut i = 0usize;
    while i + 8 <= n {
        let v = _mm256_loadu_ps(bp.add(i));
        let s = _mm256_loadu_ps(sp.add(i));
        let scaled = _mm256_mul_ps(v, scale_v);
        _mm256_storeu_ps(bp.add(i), _mm256_mul_ps(scaled, s));
        i += 8;
    }
}

/// Encode-side `√(re² + im²)` over interleaved (re, im) pairs
/// (`docs/X86_64_SIMD_PLAN.md` §4.5). Not on the query hot path —
/// the win is one-time at encode.
///
/// Note per §4.5: we use IEEE `_mm256_sqrt_ps`, NOT `vrsqrtps +
/// Newton`. The reciprocal-sqrt approximation needs a multiply by
/// the input to recover the actual magnitude, which re-introduces
/// the dependency chain that sqrt's pipelining would have hidden,
/// AND rsqrt's ~12-bit precision is below the 1e-6 absolute
/// tolerance the magnitudes test expects. The IEEE sqrt is fully
/// pipelined here (8 lanes per issue) so latency is hidden anyway.
#[target_feature(enable = "avx2,fma")]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
pub(crate) unsafe fn pair_magnitudes(pairs: &[f32], mags: &mut [f32]) {
    let p = mags.len();
    let pp = pairs.as_ptr();
    let mp = mags.as_mut_ptr();
    let mut k = 0usize;

    // ---- Main loop: 8 pairs / iter ----
    while k + 8 <= p {
        let r0 = _mm256_loadu_ps(pp.add(2 * k));
        let r1 = _mm256_loadu_ps(pp.add(2 * k + 8));
        let (re, im) = deinterleave_pairs(r0, r1);
        let re_sq = _mm256_mul_ps(re, re);
        let sum = _mm256_fmadd_ps(im, im, re_sq);
        let m = _mm256_sqrt_ps(sum);
        _mm256_storeu_ps(mp.add(k), m);
        k += 8;
    }

    // ---- 4-pair SSE cleanup ----
    if k + 4 <= p {
        let r0 = _mm_loadu_ps(pp.add(2 * k));
        let r1 = _mm_loadu_ps(pp.add(2 * k + 4));
        // SSE shuffle is already lane-local on 128 bits — no cross-
        // lane permute needed.
        let re = _mm_shuffle_ps::<0x88>(r0, r1);
        let im = _mm_shuffle_ps::<0xDD>(r0, r1);
        let re_sq = _mm_mul_ps(re, re);
        let sum = _mm_fmadd_ps(im, im, re_sq);
        let m = _mm_sqrt_ps(sum);
        _mm_storeu_ps(mp.add(k), m);
        k += 4;
    }

    // ---- Scalar tail (0..3 pairs) ----
    while k < p {
        let re = pairs[2 * k];
        let im = pairs[2 * k + 1];
        mags[k] = (re * re + im * im).sqrt();
        k += 1;
    }
}
