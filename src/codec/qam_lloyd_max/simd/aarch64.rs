//! NEON kernels for the QAM Lloyd-Max codec. Mirrors the scalar
//! references in the parent [`super`] module — all four hot paths
//! (`asymmetric_dot_pairs`, `symmetric_dot_pairs`, `block_hadamard_*`,
//! `pair_magnitudes`) here have identical signatures to the scalar fns
//! and are reached only via the parent dispatcher on aarch64 targets.
//!
//! Behavior is bit-for-bit unchanged from the previous in-file NEON
//! code; this file is the Phase 1 scaffolding split (see
//! `docs/X86_64_SIMD_PLAN.md` §5.1).

#![allow(unsafe_op_in_unsafe_fn)]

#[target_feature(enable = "neon")]
// SAFETY: requires NEON (aarch64 v8-a baseline, always present); the codec/sketch dispatcher only invokes this on aarch64.
pub(super) unsafe fn asymmetric_dot_pairs(
    q_rot: &[f32],
    amp_indices: &[u32],
    phase_indices: &[u32],
    sigma_per_pair: &[f32],
    amp_levels_unit: &[f32],
    phase_cos_lut: &[f32],
    phase_sin_lut: &[f32],
) -> (f32, f32) {
    use std::arch::aarch64::{
        float32x4_t, vaddq_f32, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vld2q_f32, vmulq_f32,
    };

    let p = amp_indices.len();

    let q_ptr = q_rot.as_ptr();
    let mut k = 0usize;

    // Stack scratch buffers for gathered values. The compiler emits
    // scalar loads per buffer, then one vld1q_f32 per 4 lanes — a
    // textbook emulated gather, fast on Apple Silicon thanks to its big
    // load width. Sized for the 8-wide main loop.
    let mut amp_buf = [0.0f32; 8];
    let mut cos_buf = [0.0f32; 8];
    let mut sin_buf = [0.0f32; 8];

    // Main loop: 8 pairs per iteration with two independent accumulator
    // pairs (lo/hi). M-series cores have 4 FP pipes with ~4-cycle FMA
    // latency, so a single dependent FMA chain stalls; two parallel
    // chains keep more pipes busy. Per lane the work is fused to
    //   tmp = q_re·cos + q_im·sin   (1 mul + 1 fma)
    //   dot += tmp·amp              (1 fma, amp deferred to one final
    //                                independent multiply)
    // which is one fewer mul and one fewer dependent FMA into `dot`
    // than the 2·(mul,fma) form.
    let mut dot0: float32x4_t = vdupq_n_f32(0.0);
    let mut dot1: float32x4_t = vdupq_n_f32(0.0);
    let mut norm0: float32x4_t = vdupq_n_f32(0.0);
    let mut norm1: float32x4_t = vdupq_n_f32(0.0);

    while k + 8 <= p {
        for j in 0..8 {
            let ai = *amp_indices.get_unchecked(k + j) as usize;
            let pi = *phase_indices.get_unchecked(k + j) as usize;
            let sigma = *sigma_per_pair.get_unchecked(k + j);
            amp_buf[j] = sigma * *amp_levels_unit.get_unchecked(ai);
            cos_buf[j] = *phase_cos_lut.get_unchecked(pi);
            sin_buf[j] = *phase_sin_lut.get_unchecked(pi);
        }

        let amp_lo = vld1q_f32(amp_buf.as_ptr());
        let amp_hi = vld1q_f32(amp_buf.as_ptr().add(4));
        let cos_lo = vld1q_f32(cos_buf.as_ptr());
        let cos_hi = vld1q_f32(cos_buf.as_ptr().add(4));
        let sin_lo = vld1q_f32(sin_buf.as_ptr());
        let sin_hi = vld1q_f32(sin_buf.as_ptr().add(4));

        // vld2q_f32 deinterleaves 8 contiguous f32s into (re, im) 4-lane
        // vectors. q_rot is (re_0, im_0, re_1, im_1, …); the lo load
        // covers pairs [k, k+4), the hi load pairs [k+4, k+8).
        let q_lo = vld2q_f32(q_ptr.add(2 * k));
        let q_hi = vld2q_f32(q_ptr.add(2 * k + 8));

        let tmp_lo = vfmaq_f32(vmulq_f32(q_lo.0, cos_lo), q_lo.1, sin_lo);
        let tmp_hi = vfmaq_f32(vmulq_f32(q_hi.0, cos_hi), q_hi.1, sin_hi);
        dot0 = vfmaq_f32(dot0, tmp_lo, amp_lo);
        dot1 = vfmaq_f32(dot1, tmp_hi, amp_hi);
        norm0 = vfmaq_f32(norm0, amp_lo, amp_lo);
        norm1 = vfmaq_f32(norm1, amp_hi, amp_hi);

        k += 8;
    }

    let mut dot_acc = vaddq_f32(dot0, dot1);
    let mut norm_acc = vaddq_f32(norm0, norm1);

    // 4-wide cleanup for a single remaining group of 4.
    while k + 4 <= p {
        for j in 0..4 {
            let ai = *amp_indices.get_unchecked(k + j) as usize;
            let pi = *phase_indices.get_unchecked(k + j) as usize;
            let sigma = *sigma_per_pair.get_unchecked(k + j);
            amp_buf[j] = sigma * *amp_levels_unit.get_unchecked(ai);
            cos_buf[j] = *phase_cos_lut.get_unchecked(pi);
            sin_buf[j] = *phase_sin_lut.get_unchecked(pi);
        }
        let amp4 = vld1q_f32(amp_buf.as_ptr());
        let cos4 = vld1q_f32(cos_buf.as_ptr());
        let sin4 = vld1q_f32(sin_buf.as_ptr());
        let q = vld2q_f32(q_ptr.add(2 * k));
        let tmp = vfmaq_f32(vmulq_f32(q.0, cos4), q.1, sin4);
        dot_acc = vfmaq_f32(dot_acc, tmp, amp4);
        norm_acc = vfmaq_f32(norm_acc, amp4, amp4);
        k += 4;
    }

    let mut dot = vaddvq_f32(dot_acc);
    let mut norm_sq = vaddvq_f32(norm_acc);

    // Tail (P not divisible by 4 — only happens for tiny test dims).
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

#[cfg(any(test, feature = "bench"))]
#[target_feature(enable = "neon")]
#[allow(clippy::too_many_arguments)]
// SAFETY: requires NEON (aarch64 v8-a baseline, always present); the codec/sketch dispatcher only invokes this on aarch64.
pub(super) unsafe fn symmetric_dot_pairs(
    amp_indices_a: &[u32],
    phase_indices_a: &[u32],
    amp_indices_b: &[u32],
    phase_indices_b: &[u32],
    sigma_per_pair: &[f32],
    amp_levels_unit: &[f32],
    phase_cos_lut: &[f32],
    phase_sin_lut: &[f32],
) -> (f32, f32, f32) {
    use std::arch::aarch64::{
        float32x4_t, vaddvq_f32, vdupq_n_f32, vfmaq_f32, vld1q_f32, vmulq_f32,
    };

    let p = amp_indices_a.len();
    let mut dot_acc: float32x4_t = vdupq_n_f32(0.0);
    let mut na_acc: float32x4_t = vdupq_n_f32(0.0);
    let mut nb_acc: float32x4_t = vdupq_n_f32(0.0);

    let mut amp_a_buf = [0.0f32; 4];
    let mut amp_b_buf = [0.0f32; 4];
    let mut cos_a_buf = [0.0f32; 4];
    let mut sin_a_buf = [0.0f32; 4];
    let mut cos_b_buf = [0.0f32; 4];
    let mut sin_b_buf = [0.0f32; 4];

    let mut k = 0usize;
    while k + 4 <= p {
        for j in 0..4 {
            let ai_a = *amp_indices_a.get_unchecked(k + j) as usize;
            let pi_a = *phase_indices_a.get_unchecked(k + j) as usize;
            let ai_b = *amp_indices_b.get_unchecked(k + j) as usize;
            let pi_b = *phase_indices_b.get_unchecked(k + j) as usize;
            let sigma = *sigma_per_pair.get_unchecked(k + j);
            amp_a_buf[j] = sigma * *amp_levels_unit.get_unchecked(ai_a);
            amp_b_buf[j] = sigma * *amp_levels_unit.get_unchecked(ai_b);
            cos_a_buf[j] = *phase_cos_lut.get_unchecked(pi_a);
            sin_a_buf[j] = *phase_sin_lut.get_unchecked(pi_a);
            cos_b_buf[j] = *phase_cos_lut.get_unchecked(pi_b);
            sin_b_buf[j] = *phase_sin_lut.get_unchecked(pi_b);
        }
        let amp_a4 = vld1q_f32(amp_a_buf.as_ptr());
        let amp_b4 = vld1q_f32(amp_b_buf.as_ptr());
        let cos_a4 = vld1q_f32(cos_a_buf.as_ptr());
        let sin_a4 = vld1q_f32(sin_a_buf.as_ptr());
        let cos_b4 = vld1q_f32(cos_b_buf.as_ptr());
        let sin_b4 = vld1q_f32(sin_b_buf.as_ptr());

        // cos(θ_a − θ_b) = cos·cos + sin·sin
        let cos_diff = vfmaq_f32(vmulq_f32(sin_a4, sin_b4), cos_a4, cos_b4);
        let amp_prod = vmulq_f32(amp_a4, amp_b4);
        dot_acc = vfmaq_f32(dot_acc, amp_prod, cos_diff);
        na_acc = vfmaq_f32(na_acc, amp_a4, amp_a4);
        nb_acc = vfmaq_f32(nb_acc, amp_b4, amp_b4);
        k += 4;
    }

    let mut dot = vaddvq_f32(dot_acc);
    let mut na_sq = vaddvq_f32(na_acc);
    let mut nb_sq = vaddvq_f32(nb_acc);

    while k < p {
        let ai_a = amp_indices_a[k] as usize;
        let pi_a = phase_indices_a[k] as usize;
        let ai_b = amp_indices_b[k] as usize;
        let pi_b = phase_indices_b[k] as usize;
        let sigma = sigma_per_pair[k];
        let amp_a = sigma * amp_levels_unit[ai_a];
        let amp_b = sigma * amp_levels_unit[ai_b];
        let cos_diff =
            phase_cos_lut[pi_a] * phase_cos_lut[pi_b] + phase_sin_lut[pi_a] * phase_sin_lut[pi_b];
        dot += amp_a * amp_b * cos_diff;
        na_sq += amp_a * amp_a;
        nb_sq += amp_b * amp_b;
        k += 1;
    }

    (dot, na_sq, nb_sq)
}

/// Butterfly stages `h = 2 .. n/2` operating in place on `block[0..n]`.
/// Shared by the fused forward and inverse kernels — only the h=1 stage
/// and the scale/sign passes differ between them.
#[target_feature(enable = "neon")]
// SAFETY: requires NEON (aarch64 v8-a baseline, always present); the codec/sketch dispatcher only invokes this on aarch64.
unsafe fn fwht_stages_h2_plus(bp: *mut f32, n: usize) {
    use std::arch::aarch64::{
        vadd_f32, vaddq_f32, vcombine_f32, vget_high_f32, vget_low_f32, vld1q_f32, vst1q_f32,
        vsub_f32, vsubq_f32,
    };

    // Stage h=2: butterfly within each group of 4.
    // For v=[a,b,c,d]: lo=(a,b), hi=(c,d) → (a+c, b+d, a-c, b-d)
    let mut i = 0usize;
    while i + 4 <= n {
        let v = vld1q_f32(bp.add(i));
        let lo = vget_low_f32(v);
        let hi = vget_high_f32(v);
        let new_lo = vadd_f32(lo, hi);
        let new_hi = vsub_f32(lo, hi);
        vst1q_f32(bp.add(i), vcombine_f32(new_lo, new_hi));
        i += 4;
    }

    // Stages h >= 4: paired vector loads at offset (i, i+h).
    let mut h = 4usize;
    while h < n {
        let mut base = 0usize;
        while base < n {
            let mut j = 0usize;
            while j < h {
                let a = vld1q_f32(bp.add(base + j));
                let b = vld1q_f32(bp.add(base + j + h));
                vst1q_f32(bp.add(base + j), vaddq_f32(a, b));
                vst1q_f32(bp.add(base + j + h), vsubq_f32(a, b));
                j += 4;
            }
            base += 2 * h;
        }
        h *= 2;
    }
}

/// Fused forward block-Hadamard: the random sign multiply is folded into
/// the h=1 butterfly stage, eliminating the separate full-array sign
/// pass. `block` and `signs` are one block.
#[target_feature(enable = "neon")]
// SAFETY: requires NEON (aarch64 v8-a baseline, always present); the codec/sketch dispatcher only invokes this on aarch64.
pub(super) unsafe fn fwht_forward_fused(block: &mut [f32], signs: &[f32]) {
    use std::arch::aarch64::{
        vaddq_f32, vld1q_f32, vmulq_f32, vmulq_n_f32, vrev64q_f32, vst1q_f32,
    };

    let n = block.len();
    debug_assert!(n.is_power_of_two() && n >= 4);
    debug_assert_eq!(signs.len(), n);
    let bp = block.as_mut_ptr();
    let sp = signs.as_ptr();

    // Stage h=1 with the sign multiply folded in: load raw data and the
    // per-element signs, form v = data·sign, then the pair butterfly
    //   result = signed(v) + rev64(v) = [a+b, a-b, c+d, c-d]
    // where signed multiplies lanes by [1, -1, 1, -1]. Identical to
    // apply-signs-then-h=1, but with one fewer pass over the block.
    let lane_signs = vld1q_f32([1.0_f32, -1.0, 1.0, -1.0].as_ptr());
    let mut i = 0usize;
    while i + 4 <= n {
        let raw = vld1q_f32(bp.add(i));
        let s = vld1q_f32(sp.add(i));
        let v = vmulq_f32(raw, s);
        let swap = vrev64q_f32(v);
        let v_signed = vmulq_f32(v, lane_signs);
        vst1q_f32(bp.add(i), vaddq_f32(v_signed, swap));
        i += 4;
    }

    fwht_stages_h2_plus(bp, n);

    // 1/sqrt(n) scaling.
    let scale = (n as f32).sqrt().recip();
    let mut i = 0usize;
    while i + 4 <= n {
        let v = vld1q_f32(bp.add(i));
        vst1q_f32(bp.add(i), vmulq_n_f32(v, scale));
        i += 4;
    }
}

/// Fused inverse block-Hadamard: FWHT first, then the final pass folds
/// the 1/sqrt(n) scale and the sign multiply into a single load·scale·
/// sign·store — one pass instead of the old scale-pass + sign-pass.
#[target_feature(enable = "neon")]
// SAFETY: requires NEON (aarch64 v8-a baseline, always present); the codec/sketch dispatcher only invokes this on aarch64.
pub(super) unsafe fn fwht_inverse_fused(block: &mut [f32], signs: &[f32]) {
    use std::arch::aarch64::{
        vaddq_f32, vld1q_f32, vmulq_f32, vmulq_n_f32, vrev64q_f32, vst1q_f32,
    };

    let n = block.len();
    debug_assert!(n.is_power_of_two() && n >= 4);
    debug_assert_eq!(signs.len(), n);
    let bp = block.as_mut_ptr();
    let sp = signs.as_ptr();

    // Stage h=1 (plain — for the inverse, signs are applied after FWHT).
    let lane_signs = vld1q_f32([1.0_f32, -1.0, 1.0, -1.0].as_ptr());
    let mut i = 0usize;
    while i + 4 <= n {
        let v = vld1q_f32(bp.add(i));
        let swap = vrev64q_f32(v);
        let v_signed = vmulq_f32(v, lane_signs);
        vst1q_f32(bp.add(i), vaddq_f32(v_signed, swap));
        i += 4;
    }

    fwht_stages_h2_plus(bp, n);

    // Final pass: data[i] := data[i] · (1/sqrt(n)) · signs[i].
    let scale = (n as f32).sqrt().recip();
    let mut i = 0usize;
    while i + 4 <= n {
        let v = vld1q_f32(bp.add(i));
        let s = vld1q_f32(sp.add(i));
        let scaled = vmulq_n_f32(v, scale);
        vst1q_f32(bp.add(i), vmulq_f32(scaled, s));
        i += 4;
    }
}

#[target_feature(enable = "neon")]
// SAFETY: requires NEON (aarch64 v8-a baseline, always present); the codec/sketch dispatcher only invokes this on aarch64.
pub(super) unsafe fn pair_magnitudes(pairs: &[f32], mags: &mut [f32]) {
    use std::arch::aarch64::{vfmaq_f32, vld2q_f32, vmulq_f32, vsqrtq_f32, vst1q_f32};
    let p = mags.len();
    let pp = pairs.as_ptr();
    let mp = mags.as_mut_ptr();
    let mut k = 0usize;
    while k + 4 <= p {
        let q = vld2q_f32(pp.add(2 * k));
        let re = q.0;
        let im = q.1;
        let re_sq = vmulq_f32(re, re);
        let sum = vfmaq_f32(re_sq, im, im);
        let m = vsqrtq_f32(sum);
        vst1q_f32(mp.add(k), m);
        k += 4;
    }
    while k < p {
        let re = pairs[2 * k];
        let im = pairs[2 * k + 1];
        mags[k] = (re * re + im * im).sqrt();
        k += 1;
    }
}
