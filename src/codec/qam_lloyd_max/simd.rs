//! SIMD kernels for the QAM Lloyd-Max codec.
//!
//! Three hot-path kernels:
//!
//! 1. [`asymmetric_dot_pairs`] — production-critical asymmetric distance
//!    inner loop (the search hot path). Processes 8 complex pairs per
//!    iteration with two independent accumulator pairs, gathering
//!    `sigma · level[ai]`, `cos[pi]`, `sin[pi]` and reducing into the
//!    dot product and decoded-norm-squared accumulators. This kernel is
//!    gather-bound (≈5–6 scalar LUT loads/pair), so the unroll buys less
//!    than an FMA-bound loop would.
//!
//! 2. [`block_hadamard_forward`] / [`block_hadamard_inverse`] — fused
//!    sign multiply + per-block normalized FWHT. Used by encode,
//!    decode, prepare_query.
//!
//! 3. [`pair_magnitudes`] — vectorized `√(re² + im²)` over interleaved
//!    `(re_0, im_0, re_1, im_1, …)`. Cuts encode-time scalar overhead
//!    on the magnitude side; scalar `atan2` stays in place (no native
//!    instruction, tunable polynomial doesn't beat libm by much).
//!
//! Each kernel has a per-arch SIMD implementation (NEON on aarch64,
//! AVX2 + FMA on x86_64) in a sibling module under `simd/`, and a
//! portable scalar fallback in this file. Both SIMD paths produce
//! identical output to the scalar reference up to fp associativity (the
//! NEON path uses two parallel accumulators, so reduction order differs
//! — within ~1 ULP of the scalar reference on f32).
//!
//! ## Reference layout
//!
//! This `<module>/simd.rs` + `<module>/simd/<arch>.rs` shape is the
//! canonical per-arch SIMD layout for the crate (see the "Per-arch SIMD"
//! rule in `SKILL.md`); `codec/qam_sliding/` and `retrieval/sketch/`
//! mirror it. Here the dispatch lives in `simd.rs` itself, so the arch
//! submodules can stay private; the other two keep their dispatch in the
//! parent module and so declare the arch submodules `pub(super)`.

#[cfg(target_arch = "aarch64")]
mod aarch64;
#[cfg(target_arch = "x86_64")]
mod x86_64;

// ============================================================
// asymmetric_dot_pairs
// ============================================================

/// Compute `(dot, y_hat_norm_sq)` for the asymmetric distance kernel.
///
/// ```text
/// for k in 0..P:
///     amp = sigma_per_pair[k] · amp_levels_unit[amp_indices[k]]
///     c   = phase_cos_lut[phase_indices[k]]
///     s   = phase_sin_lut[phase_indices[k]]
///     dot         += q_rot[2k]   · amp · c + q_rot[2k+1] · amp · s
///     y_hat_norm² += amp · amp
/// ```
///
/// Caller has already validated index bounds.
pub(crate) fn asymmetric_dot_pairs(
    q_rot: &[f32],
    amp_indices: &[u32],
    phase_indices: &[u32],
    sigma_per_pair: &[f32],
    amp_levels_unit: &[f32],
    phase_cos_lut: &[f32],
    phase_sin_lut: &[f32],
) -> (f32, f32) {
    debug_assert_eq!(q_rot.len(), 2 * amp_indices.len());
    debug_assert_eq!(amp_indices.len(), phase_indices.len());
    debug_assert_eq!(sigma_per_pair.len(), amp_indices.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return aarch64::asymmetric_dot_pairs(
            q_rot,
            amp_indices,
            phase_indices,
            sigma_per_pair,
            amp_levels_unit,
            phase_cos_lut,
            phase_sin_lut,
        );
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        // Static dispatch: when the crate is built with
        // `-C target-cpu=x86-64-v3` (or the `x86_64_baseline_v3`
        // Cargo feature, or any `target-cpu` that implies avx2+fma),
        // the AVX2 path is statically known to be available — no
        // runtime check, no dispatcher call boundary (the compiler
        // can inline `x86_64::asymmetric_dot_pairs` straight into
        // the caller since both share the target_feature context).
        return x86_64::asymmetric_dot_pairs(
            q_rot,
            amp_indices,
            phase_indices,
            sigma_per_pair,
            amp_levels_unit,
            phase_cos_lut,
            phase_sin_lut,
        );
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx2", target_feature = "fma"))
    ))]
    {
        // Runtime dispatch: portable build, supports any x86_64.
        // The CPU feature query is cached behind a single
        // `OnceLock<bool>` after the first call.
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                return x86_64::asymmetric_dot_pairs(
                    q_rot,
                    amp_indices,
                    phase_indices,
                    sigma_per_pair,
                    amp_levels_unit,
                    phase_cos_lut,
                    phase_sin_lut,
                );
            }
        }
    }
    #[allow(unreachable_code)]
    scalar_asymmetric_dot_pairs(
        q_rot,
        amp_indices,
        phase_indices,
        sigma_per_pair,
        amp_levels_unit,
        phase_cos_lut,
        phase_sin_lut,
    )
}

/// Reference scalar implementation. Always available — used as a
/// non-aarch64 fallback and as the test ground-truth on aarch64.
pub(crate) fn scalar_asymmetric_dot_pairs(
    q_rot: &[f32],
    amp_indices: &[u32],
    phase_indices: &[u32],
    sigma_per_pair: &[f32],
    amp_levels_unit: &[f32],
    phase_cos_lut: &[f32],
    phase_sin_lut: &[f32],
) -> (f32, f32) {
    let p = amp_indices.len();
    let mut dot: f32 = 0.0;
    let mut norm_sq: f32 = 0.0;
    for k in 0..p {
        let ai = amp_indices[k] as usize;
        let pi = phase_indices[k] as usize;
        let amp = sigma_per_pair[k] * amp_levels_unit[ai];
        let c = phase_cos_lut[pi];
        let s = phase_sin_lut[pi];
        let q_re = q_rot[2 * k];
        let q_im = q_rot[2 * k + 1];
        dot += q_re * amp * c + q_im * amp * s;
        norm_sq += amp * amp;
    }
    (dot, norm_sq)
}

// ============================================================
// symmetric_dot_pairs — code-level same-codec inner product
// ============================================================

/// Compute `(dot, norm_a_sq, norm_b_sq)` for two QAM-encoded vectors
/// without ever decoding back to f32. The math (rotation-invariant):
///
/// ```text
/// <y_hat_a, y_hat_b> = Σ_k amp_a · amp_b · cos(θ_a - θ_b)
///                    = Σ_k amp_a · amp_b · (cos·cos + sin·sin)
/// ```
///
/// where `amp = σ_k · L[ai]` and `(cos, sin) = (cos[pi], sin[pi])`.
/// Reachable only through the public `symmetric_distance` wrapper, which
/// today is exercised only by tests and the `bench` adapter — not by the
/// production search or calibration paths. Kept for parity with the
/// asymmetric kernel; it is intentionally left in the plain 4-wide form
/// since it is not a shipped hot path.
///
/// Caller has already validated index bounds.
#[cfg(any(test, feature = "bench"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn symmetric_dot_pairs(
    amp_indices_a: &[u32],
    phase_indices_a: &[u32],
    amp_indices_b: &[u32],
    phase_indices_b: &[u32],
    sigma_per_pair: &[f32],
    amp_levels_unit: &[f32],
    phase_cos_lut: &[f32],
    phase_sin_lut: &[f32],
) -> (f32, f32, f32) {
    debug_assert_eq!(amp_indices_a.len(), phase_indices_a.len());
    debug_assert_eq!(amp_indices_a.len(), amp_indices_b.len());
    debug_assert_eq!(amp_indices_b.len(), phase_indices_b.len());
    debug_assert_eq!(sigma_per_pair.len(), amp_indices_a.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return aarch64::symmetric_dot_pairs(
            amp_indices_a,
            phase_indices_a,
            amp_indices_b,
            phase_indices_b,
            sigma_per_pair,
            amp_levels_unit,
            phase_cos_lut,
            phase_sin_lut,
        );
    }
    #[allow(unreachable_code)]
    scalar_symmetric_dot_pairs(
        amp_indices_a,
        phase_indices_a,
        amp_indices_b,
        phase_indices_b,
        sigma_per_pair,
        amp_levels_unit,
        phase_cos_lut,
        phase_sin_lut,
    )
}

#[cfg(any(test, feature = "bench"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn scalar_symmetric_dot_pairs(
    amp_indices_a: &[u32],
    phase_indices_a: &[u32],
    amp_indices_b: &[u32],
    phase_indices_b: &[u32],
    sigma_per_pair: &[f32],
    amp_levels_unit: &[f32],
    phase_cos_lut: &[f32],
    phase_sin_lut: &[f32],
) -> (f32, f32, f32) {
    let p = amp_indices_a.len();
    let mut dot: f32 = 0.0;
    let mut na_sq: f32 = 0.0;
    let mut nb_sq: f32 = 0.0;
    for k in 0..p {
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
    }
    (dot, na_sq, nb_sq)
}

// ============================================================
// block_hadamard_forward / inverse
// ============================================================

/// In-place block-Hadamard forward: `data := H · data` where
/// `H = sign-mul · FWHT_per_block`. `data.len()` must be a multiple of
/// `block_size` and `block_size` must be a power of two.
pub(crate) fn block_hadamard_forward(data: &mut [f32], signs: &[f32], block_size: usize) {
    debug_assert_eq!(data.len(), signs.len());
    debug_assert!(block_size.is_power_of_two() && block_size > 0);
    debug_assert!(data.len().is_multiple_of(block_size));

    #[cfg(target_arch = "aarch64")]
    {
        if block_size >= 4 {
            for (chunk, sign_chunk) in data
                .chunks_exact_mut(block_size)
                .zip(signs.chunks_exact(block_size))
            {
                unsafe { aarch64::fwht_forward_fused(chunk, sign_chunk) };
            }
            return;
        }
    }
    // §4.4 dispatcher guard: block_size >= 8 (the AVX2 h=1 stage
    // loads 8 floats; block_size = 4 would OOB). Smaller falls to
    // scalar; the parity test for block_size = 4 still passes
    // because the dispatcher routes it there.
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        if block_size >= 8 {
            for (chunk, sign_chunk) in data
                .chunks_exact_mut(block_size)
                .zip(signs.chunks_exact(block_size))
            {
                unsafe { x86_64::fwht_forward_fused(chunk, sign_chunk) };
            }
            return;
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx2", target_feature = "fma"))
    ))]
    {
        if block_size >= 8
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            for (chunk, sign_chunk) in data
                .chunks_exact_mut(block_size)
                .zip(signs.chunks_exact(block_size))
            {
                unsafe { x86_64::fwht_forward_fused(chunk, sign_chunk) };
            }
            return;
        }
    }
    scalar_block_hadamard_forward(data, signs, block_size);
}

/// In-place block-Hadamard inverse: `data := H · data` (`H` self-inverse).
/// Inverse means we apply FWHT first and then sign multiply.
pub(crate) fn block_hadamard_inverse(data: &mut [f32], signs: &[f32], block_size: usize) {
    debug_assert_eq!(data.len(), signs.len());
    debug_assert!(block_size.is_power_of_two() && block_size > 0);
    debug_assert!(data.len().is_multiple_of(block_size));

    #[cfg(target_arch = "aarch64")]
    {
        if block_size >= 4 {
            for (chunk, sign_chunk) in data
                .chunks_exact_mut(block_size)
                .zip(signs.chunks_exact(block_size))
            {
                unsafe { aarch64::fwht_inverse_fused(chunk, sign_chunk) };
            }
            return;
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    {
        if block_size >= 8 {
            for (chunk, sign_chunk) in data
                .chunks_exact_mut(block_size)
                .zip(signs.chunks_exact(block_size))
            {
                unsafe { x86_64::fwht_inverse_fused(chunk, sign_chunk) };
            }
            return;
        }
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx2", target_feature = "fma"))
    ))]
    {
        if block_size >= 8
            && std::is_x86_feature_detected!("avx2")
            && std::is_x86_feature_detected!("fma")
        {
            for (chunk, sign_chunk) in data
                .chunks_exact_mut(block_size)
                .zip(signs.chunks_exact(block_size))
            {
                unsafe { x86_64::fwht_inverse_fused(chunk, sign_chunk) };
            }
            return;
        }
    }
    scalar_block_hadamard_inverse(data, signs, block_size);
}

pub(crate) fn scalar_block_hadamard_forward(data: &mut [f32], signs: &[f32], block_size: usize) {
    for (d, s) in data.iter_mut().zip(signs.iter()) {
        *d *= *s;
    }
    scalar_fwht_block_normalized(data, block_size);
}

pub(crate) fn scalar_block_hadamard_inverse(data: &mut [f32], signs: &[f32], block_size: usize) {
    scalar_fwht_block_normalized(data, block_size);
    for (d, s) in data.iter_mut().zip(signs.iter()) {
        *d *= *s;
    }
}

fn scalar_fwht_block_normalized(data: &mut [f32], block_size: usize) {
    let scale = (block_size as f32).sqrt().recip();
    for block in data.chunks_exact_mut(block_size) {
        let n = block.len();
        let mut h = 1usize;
        while h < n {
            let mut i = 0usize;
            while i < n {
                for j in i..i + h {
                    let x = block[j];
                    let y = block[j + h];
                    block[j] = x + y;
                    block[j + h] = x - y;
                }
                i += 2 * h;
            }
            h *= 2;
        }
        for v in block.iter_mut() {
            *v *= scale;
        }
    }
}

// ============================================================
// pair_magnitudes — encode-side `√(re² + im²)`
// ============================================================

/// In-place: `mags[k] := sqrt(pairs[2k]² + pairs[2k+1]²)` for k in
/// 0..mags.len(). `pairs.len()` must be `2 · mags.len()`.
pub(crate) fn pair_magnitudes(pairs: &[f32], mags: &mut [f32]) {
    debug_assert_eq!(pairs.len(), 2 * mags.len());

    #[cfg(target_arch = "aarch64")]
    unsafe {
        return aarch64::pair_magnitudes(pairs, mags);
    }
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "fma"
    ))]
    unsafe {
        return x86_64::pair_magnitudes(pairs, mags);
    }
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx2", target_feature = "fma"))
    ))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma") {
            unsafe {
                return x86_64::pair_magnitudes(pairs, mags);
            }
        }
    }
    #[allow(unreachable_code)]
    scalar_pair_magnitudes(pairs, mags)
}

pub(crate) fn scalar_pair_magnitudes(pairs: &[f32], mags: &mut [f32]) {
    for (k, m) in mags.iter_mut().enumerate() {
        let re = pairs[2 * k];
        let im = pairs[2 * k + 1];
        *m = (re * re + im * im).sqrt();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::prng::SplitMix64;

    fn random_unit_vector(dim: usize, seed: u64) -> Vec<f32> {
        let mut prng = SplitMix64::new(seed);
        let mut v = vec![0.0f32; dim];
        let mut i = 0;
        while i + 1 < dim {
            // Box-Muller from two uniform draws in (0, 1).
            let u1 = (prng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let u2 = (prng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            let u1 = u1.max(1e-300);
            let r = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            v[i] = (r * theta.cos()) as f32;
            v[i + 1] = (r * theta.sin()) as f32;
            i += 2;
        }
        let n2: f32 = v.iter().map(|&x| x * x).sum();
        let inv = 1.0 / n2.sqrt();
        for x in v.iter_mut() {
            *x *= inv;
        }
        v
    }

    fn deterministic_signs(dim: usize, seed: u64) -> Vec<f32> {
        let mut prng = SplitMix64::new(seed);
        let mut s = Vec::with_capacity(dim);
        let mut bits = 0u64;
        let mut bits_left = 0u32;
        for _ in 0..dim {
            if bits_left == 0 {
                bits = prng.next_u64();
                bits_left = 64;
            }
            s.push(if bits & 1 == 0 { 1.0_f32 } else { -1.0_f32 });
            bits >>= 1;
            bits_left -= 1;
        }
        s
    }

    fn random_indices(p: usize, max_val: u32, seed: u64) -> Vec<u32> {
        let mut prng = SplitMix64::new(seed);
        (0..p).map(|_| (prng.next_u64() as u32) % max_val).collect()
    }

    fn random_floats(n: usize, seed: u64) -> Vec<f32> {
        let mut prng = SplitMix64::new(seed);
        (0..n)
            .map(|_| {
                let u = (prng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
                (u * 2.0 - 1.0) as f32
            })
            .collect()
    }

    // ============================================================
    // asymmetric_dot_pairs
    // ============================================================

    #[test]
    fn asymmetric_dot_pairs_simd_matches_scalar_p1536() {
        let p = 1536;
        let dim = 2 * p;
        let n_amp = 32;
        let n_phase = 64;
        let q_rot = random_unit_vector(dim, 0x1111);
        let amp_idx = random_indices(p, n_amp, 0x2222);
        let phase_idx = random_indices(p, n_phase, 0x3333);
        let sigma: Vec<f32> = random_floats(p, 0x4444)
            .into_iter()
            .map(|v| (v.abs() + 0.01) / 32.0)
            .collect();
        let levels: Vec<f32> = (0..n_amp)
            .map(|i| (i as f32 + 0.5) / n_amp as f32 * 4.0)
            .collect();
        let cos_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).cos())
            .collect();
        let sin_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).sin())
            .collect();

        let (dot_simd, norm_simd) = asymmetric_dot_pairs(
            &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut,
        );
        let (dot_scalar, norm_scalar) = scalar_asymmetric_dot_pairs(
            &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut,
        );

        let dot_rel = (dot_simd - dot_scalar).abs() / (dot_scalar.abs().max(1e-6));
        let norm_rel = (norm_simd - norm_scalar).abs() / (norm_scalar.abs().max(1e-6));
        // 2e-4 relative — §11 decision rule for AVX2's wider 16-way
        // reduction (NEON's 8-way passes at 1e-4 with headroom; AVX2
        // needs the slightly looser bound). Both arches use the same
        // dispatcher-level test so we set the bound at the wider of
        // the two reduction shapes.
        assert!(
            dot_rel < 2e-4,
            "dot drift: simd={dot_simd}, scalar={dot_scalar}, rel={dot_rel}"
        );
        assert!(
            norm_rel < 2e-4,
            "norm drift: simd={norm_simd}, scalar={norm_scalar}, rel={norm_rel}"
        );
    }

    #[test]
    fn asymmetric_dot_pairs_handles_tail() {
        // P = 7 → one 4-wide iteration + 3-pair tail.
        let p = 7;
        let dim = 2 * p;
        let n_amp = 8;
        let n_phase = 16;
        let q_rot = random_unit_vector(dim, 0x5555);
        let amp_idx = random_indices(p, n_amp, 0x6666);
        let phase_idx = random_indices(p, n_phase, 0x7777);
        let sigma = vec![0.05_f32; p];
        let levels: Vec<f32> = (0..n_amp).map(|i| 0.1 * i as f32).collect();
        let cos_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).cos())
            .collect();
        let sin_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).sin())
            .collect();

        let (d1, n1) = asymmetric_dot_pairs(
            &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut,
        );
        let (d2, n2) = scalar_asymmetric_dot_pairs(
            &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut,
        );
        assert!((d1 - d2).abs() < 1e-5);
        assert!((n1 - n2).abs() < 1e-5);
    }

    #[test]
    fn asymmetric_dot_pairs_zero_amp_has_zero_norm_and_dot() {
        let p = 64;
        let dim = 2 * p;
        let q_rot = random_unit_vector(dim, 0x9999);
        let amp_idx = vec![0u32; p];
        let phase_idx = vec![3u32; p];
        let sigma = vec![0.05_f32; p];
        // levels[0] = 0 → amp = 0 → both accumulators zero.
        let levels = vec![0.0_f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let cos_lut: Vec<f32> = (0..16)
            .map(|i| (i as f32 * std::f32::consts::TAU / 16.0).cos())
            .collect();
        let sin_lut: Vec<f32> = (0..16)
            .map(|i| (i as f32 * std::f32::consts::TAU / 16.0).sin())
            .collect();
        let (d, n) = asymmetric_dot_pairs(
            &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut,
        );
        assert!(d.abs() < 1e-7, "dot = {d}");
        assert!(n.abs() < 1e-7, "norm_sq = {n}");
    }

    // ============================================================
    // block_hadamard_*
    // ============================================================

    #[test]
    fn block_hadamard_simd_matches_scalar_b1024() {
        let dim = 3072;
        let block_size = 1024;
        let mut data_simd = random_unit_vector(dim, 0xA1);
        let signs = deterministic_signs(dim, 0xB2);
        let mut data_scalar = data_simd.clone();
        block_hadamard_forward(&mut data_simd, &signs, block_size);
        scalar_block_hadamard_forward(&mut data_scalar, &signs, block_size);
        for (i, (a, b)) in data_simd.iter().zip(data_scalar.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "fwd drift at {i}: simd={a}, scalar={b}"
            );
        }

        // Inverse parity.
        let mut inv_simd = data_simd.clone();
        let mut inv_scalar = data_scalar.clone();
        block_hadamard_inverse(&mut inv_simd, &signs, block_size);
        scalar_block_hadamard_inverse(&mut inv_scalar, &signs, block_size);
        for (a, b) in inv_simd.iter().zip(inv_scalar.iter()) {
            assert!((a - b).abs() < 1e-4);
        }
    }

    #[test]
    fn block_hadamard_round_trip() {
        let dim = 2048;
        let block_size = 512;
        let mut data = random_unit_vector(dim, 0xC3);
        let original = data.clone();
        let signs = deterministic_signs(dim, 0xD4);
        block_hadamard_forward(&mut data, &signs, block_size);
        block_hadamard_inverse(&mut data, &signs, block_size);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 5e-5, "round-trip drift: {a} vs {b}");
        }
    }

    #[test]
    fn block_hadamard_preserves_norm() {
        let dim = 1024;
        let block_size = 1024;
        let mut data = random_unit_vector(dim, 0xE5);
        let n_before: f32 = data.iter().map(|v| v * v).sum();
        let signs = deterministic_signs(dim, 0xF6);
        block_hadamard_forward(&mut data, &signs, block_size);
        let n_after: f32 = data.iter().map(|v| v * v).sum();
        assert!((n_before - n_after).abs() / n_before < 1e-5);
    }

    #[test]
    fn block_hadamard_block_size_4_simd_matches_scalar() {
        // Smallest NEON block_size we support — exercises the s=1 + s=2
        // shuffle stages without any s≥4 stage.
        let dim = 64;
        let block_size = 4;
        let mut a = random_unit_vector(dim, 0x77);
        let signs = deterministic_signs(dim, 0x88);
        let mut b = a.clone();
        block_hadamard_forward(&mut a, &signs, block_size);
        scalar_block_hadamard_forward(&mut b, &signs, block_size);
        for (xa, xb) in a.iter().zip(b.iter()) {
            assert!((xa - xb).abs() < 1e-5);
        }
    }

    #[test]
    fn block_hadamard_block_size_2_uses_scalar_path() {
        // block_size = 2 < 4 → falls back to scalar in the dispatcher.
        let block_size = 2;
        let mut a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let signs = vec![1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0];
        let mut b = a.clone();
        block_hadamard_forward(&mut a, &signs, block_size);
        scalar_block_hadamard_forward(&mut b, &signs, block_size);
        assert_eq!(a, b);
    }

    // ============================================================
    // pair_magnitudes
    // ============================================================

    #[test]
    fn pair_magnitudes_simd_matches_scalar() {
        let p = 1024;
        let pairs = random_floats(2 * p, 0x9A);
        let mut m_simd = vec![0.0_f32; p];
        let mut m_scalar = vec![0.0_f32; p];
        pair_magnitudes(&pairs, &mut m_simd);
        scalar_pair_magnitudes(&pairs, &mut m_scalar);
        for (a, b) in m_simd.iter().zip(m_scalar.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn pair_magnitudes_handles_tail() {
        // p = 5 → one 4-wide iteration + 1 tail pair.
        let pairs = vec![3.0, 4.0, 5.0, 12.0, 8.0, 6.0, 7.0, 24.0, 1.0, 0.0];
        let mut m = vec![0.0_f32; 5];
        pair_magnitudes(&pairs, &mut m);
        assert!((m[0] - 5.0).abs() < 1e-6);
        assert!((m[1] - 13.0).abs() < 1e-6);
        assert!((m[2] - 10.0).abs() < 1e-6);
        assert!((m[3] - 25.0).abs() < 1e-6);
        assert!((m[4] - 1.0).abs() < 1e-6);
    }

    // ============================================================
    // micro-benchmarks
    //
    // Not assertions — wall-clock timings for the hot kernels.
    // Run them with:
    //
    //   cargo test --release -p valise --lib -- --ignored --nocapture bench_
    //
    // Each is `#[ignore]`d so the normal `cargo test` run skips them.
    // ============================================================

    #[allow(clippy::print_stdout)]
    fn bench_report(name: &str, iters: u64, total: std::time::Duration, units: u64, unit: &str) {
        let ns_call = total.as_nanos() as f64 / iters as f64;
        let ns_unit = total.as_nanos() as f64 / (iters as f64 * units as f64);
        println!(
            "  {name:<42} {ns_call:>10.1} ns/call   {ns_unit:>7.3} ns/{unit}  [{iters} iters]"
        );
    }

    /// Build the representative P=1536 asymmetric inputs (n_amp=32,
    /// n_phase=64) once, matching `asymmetric_dot_pairs_simd_matches_scalar_p1536`.
    fn asym_inputs() -> (
        Vec<f32>,
        Vec<u32>,
        Vec<u32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
        Vec<f32>,
    ) {
        let p = 1536;
        let dim = 2 * p;
        let n_amp = 32;
        let n_phase = 64;
        let q_rot = random_unit_vector(dim, 0x1111);
        let amp_idx = random_indices(p, n_amp, 0x2222);
        let phase_idx = random_indices(p, n_phase, 0x3333);
        let sigma: Vec<f32> = random_floats(p, 0x4444)
            .into_iter()
            .map(|v| (v.abs() + 0.01) / 32.0)
            .collect();
        let levels: Vec<f32> = (0..n_amp)
            .map(|i| (i as f32 + 0.5) / n_amp as f32 * 4.0)
            .collect();
        let cos_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).cos())
            .collect();
        let sin_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).sin())
            .collect();
        (q_rot, amp_idx, phase_idx, sigma, levels, cos_lut, sin_lut)
    }

    #[test]
    #[ignore = "perf: cargo test --release --lib -- --ignored --nocapture bench_"]
    fn bench_asymmetric_dot_pairs() {
        use std::hint::black_box;
        let (q_rot, amp_idx, phase_idx, sigma, levels, cos_lut, sin_lut) = asym_inputs();
        let p = amp_idx.len() as u64;

        let mut sink = 0.0f32;
        for _ in 0..4_000 {
            let (d, n) = asymmetric_dot_pairs(
                &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut,
            );
            sink += d + n;
        }

        let iters: u64 = 300_000;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            let (d, n) = asymmetric_dot_pairs(
                black_box(&q_rot),
                black_box(&amp_idx),
                black_box(&phase_idx),
                black_box(&sigma),
                black_box(&levels),
                black_box(&cos_lut),
                black_box(&sin_lut),
            );
            sink += d + n;
        }
        let el = t.elapsed();
        black_box(sink);
        bench_report("asymmetric_dot_pairs P=1536", iters, el, p, "pair");
    }

    #[test]
    #[ignore = "perf: cargo test --release --lib -- --ignored --nocapture bench_"]
    fn bench_block_hadamard_forward() {
        use std::hint::black_box;
        // calibrate()/encode pass a single dim-length vector; b=1024.
        let dim = 3072;
        let block = 1024;
        let mut data = random_unit_vector(dim, 0xA1);
        let signs = deterministic_signs(dim, 0xB2);

        // FWHT is orthogonal (norm-preserving), so repeated forward
        // application stays bounded on the unit sphere — no re-init,
        // no memcpy pollution in the timed loop.
        for _ in 0..4_000 {
            block_hadamard_forward(&mut data, &signs, block);
        }
        let iters: u64 = 200_000;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            block_hadamard_forward(black_box(&mut data), black_box(&signs), black_box(block));
        }
        let el = t.elapsed();
        black_box(&data);
        bench_report(
            "block_hadamard_forward dim=3072 b=1024",
            iters,
            el,
            dim as u64,
            "elem",
        );
    }

    #[test]
    #[ignore = "perf: cargo test --release --lib -- --ignored --nocapture bench_"]
    fn bench_block_hadamard_inverse() {
        use std::hint::black_box;
        let dim = 3072;
        let block = 1024;
        let mut data = random_unit_vector(dim, 0xA1);
        let signs = deterministic_signs(dim, 0xB2);

        for _ in 0..4_000 {
            block_hadamard_inverse(&mut data, &signs, block);
        }
        let iters: u64 = 200_000;
        let t = std::time::Instant::now();
        for _ in 0..iters {
            block_hadamard_inverse(black_box(&mut data), black_box(&signs), black_box(block));
        }
        let el = t.elapsed();
        black_box(&data);
        bench_report(
            "block_hadamard_inverse dim=3072 b=1024",
            iters,
            el,
            dim as u64,
            "elem",
        );
    }

    // ============================================================
    // AVX2 asymmetric_dot_pairs parity tests (Phase 3, §4.1 / §8.1)
    //
    // These exercise the AVX2 path DIRECTLY (not through the runtime
    // dispatcher) so they fail loudly if a regression in the kernel
    // sneaks past the dispatcher test. The dispatcher tests above
    // also exercise AVX2 on x86_64 hosts but they fall to scalar on
    // hosts without AVX2.
    // ============================================================

    #[cfg(target_arch = "x86_64")]
    fn avx2_fma_available() -> bool {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }

    #[cfg(target_arch = "x86_64")]
    #[allow(clippy::too_many_arguments)]
    fn check_avx2_parity(
        q_rot: &[f32],
        amp_idx: &[u32],
        phase_idx: &[u32],
        sigma: &[f32],
        levels: &[f32],
        cos_lut: &[f32],
        sin_lut: &[f32],
        tol_rel: f32,
        case: &str,
    ) {
        if !avx2_fma_available() {
            eprintln!("skip: cpu lacks avx2+fma ({case})");
            return;
        }
        let (d_avx, n_avx) = unsafe {
            super::x86_64::asymmetric_dot_pairs(
                q_rot, amp_idx, phase_idx, sigma, levels, cos_lut, sin_lut,
            )
        };
        let (d_sc, n_sc) =
            scalar_asymmetric_dot_pairs(q_rot, amp_idx, phase_idx, sigma, levels, cos_lut, sin_lut);
        let dr = (d_avx - d_sc).abs() / (d_sc.abs().max(1e-6));
        let nr = (n_avx - n_sc).abs() / (n_sc.abs().max(1e-6));
        assert!(
            dr < tol_rel,
            "{case}: dot drift avx={d_avx} scalar={d_sc} rel={dr}"
        );
        assert!(
            nr < tol_rel,
            "{case}: norm drift avx={n_avx} scalar={n_sc} rel={nr}"
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_asymmetric_matches_scalar_p1536() {
        let p = 1536;
        let dim = 2 * p;
        let n_amp = 32;
        let n_phase = 64;
        let q_rot = random_unit_vector(dim, 0xAB_CD_01);
        let amp_idx = random_indices(p, n_amp, 0xAB_CD_02);
        let phase_idx = random_indices(p, n_phase, 0xAB_CD_03);
        let sigma: Vec<f32> = random_floats(p, 0xAB_CD_04)
            .into_iter()
            .map(|v| (v.abs() + 0.01) / 32.0)
            .collect();
        let levels: Vec<f32> = (0..n_amp)
            .map(|i| (i as f32 + 0.5) / n_amp as f32 * 4.0)
            .collect();
        let cos_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).cos())
            .collect();
        let sin_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).sin())
            .collect();
        check_avx2_parity(
            &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut, 2e-4, "p=1536",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_asymmetric_matches_scalar_p384() {
        let p = 384;
        let dim = 2 * p;
        let n_amp = 32;
        let n_phase = 64;
        let q_rot = random_unit_vector(dim, 0xBE_EF_01);
        let amp_idx = random_indices(p, n_amp, 0xBE_EF_02);
        let phase_idx = random_indices(p, n_phase, 0xBE_EF_03);
        let sigma: Vec<f32> = random_floats(p, 0xBE_EF_04)
            .into_iter()
            .map(|v| (v.abs() + 0.01) / 32.0)
            .collect();
        let levels: Vec<f32> = (0..n_amp)
            .map(|i| (i as f32 + 0.5) / n_amp as f32 * 4.0)
            .collect();
        let cos_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).cos())
            .collect();
        let sin_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).sin())
            .collect();
        check_avx2_parity(
            &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut, 2e-4, "p=384",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_asymmetric_matches_scalar_p64() {
        // p=64 → exactly 4 main-loop iterations, no 8-wide cleanup,
        // no SSE tail, no scalar tail. Exercises the hot path.
        let p = 64;
        let dim = 2 * p;
        let n_amp = 32;
        let n_phase = 64;
        let q_rot = random_unit_vector(dim, 0xC0_FE_01);
        let amp_idx = random_indices(p, n_amp, 0xC0_FE_02);
        let phase_idx = random_indices(p, n_phase, 0xC0_FE_03);
        let sigma: Vec<f32> = random_floats(p, 0xC0_FE_04)
            .into_iter()
            .map(|v| (v.abs() + 0.01) / 32.0)
            .collect();
        let levels: Vec<f32> = (0..n_amp)
            .map(|i| (i as f32 + 0.5) / n_amp as f32 * 4.0)
            .collect();
        let cos_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).cos())
            .collect();
        let sin_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).sin())
            .collect();
        check_avx2_parity(
            &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut, 2e-4, "p=64",
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_asymmetric_exercises_all_tails() {
        // p=7 → 0 main-loop iters, 0 8-wide cleanup, 1 SSE 4-wide
        // iter, 3 scalar tail iters. Covers the longest tail path.
        let p = 7;
        let dim = 2 * p;
        let n_amp = 8;
        let n_phase = 16;
        let q_rot = random_unit_vector(dim, 0xDE_AD_01);
        let amp_idx = random_indices(p, n_amp, 0xDE_AD_02);
        let phase_idx = random_indices(p, n_phase, 0xDE_AD_03);
        let sigma = vec![0.05_f32; p];
        let levels: Vec<f32> = (0..n_amp).map(|i| 0.1 * i as f32).collect();
        let cos_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).cos())
            .collect();
        let sin_lut: Vec<f32> = (0..n_phase)
            .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).sin())
            .collect();
        // Tail-only inputs (small p, deterministic sigma) need a
        // tighter bound — 1e-5 abs since `dot_scalar` may be tiny.
        if !avx2_fma_available() {
            return;
        }
        let (d_avx, n_avx) = unsafe {
            super::x86_64::asymmetric_dot_pairs(
                &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut,
            )
        };
        let (d_sc, n_sc) = scalar_asymmetric_dot_pairs(
            &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut,
        );
        assert!((d_avx - d_sc).abs() < 1e-5, "tail dot: {d_avx} vs {d_sc}");
        assert!((n_avx - n_sc).abs() < 1e-5, "tail norm: {n_avx} vs {n_sc}");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_asymmetric_zero_amp_zero_outputs() {
        let p = 64;
        let dim = 2 * p;
        let q_rot = random_unit_vector(dim, 0xFA_CE_01);
        let amp_idx = vec![0u32; p];
        let phase_idx = vec![3u32; p];
        let sigma = vec![0.05_f32; p];
        let levels = vec![0.0_f32, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5];
        let cos_lut: Vec<f32> = (0..16)
            .map(|i| (i as f32 * std::f32::consts::TAU / 16.0).cos())
            .collect();
        let sin_lut: Vec<f32> = (0..16)
            .map(|i| (i as f32 * std::f32::consts::TAU / 16.0).sin())
            .collect();
        if !avx2_fma_available() {
            return;
        }
        let (d, n) = unsafe {
            super::x86_64::asymmetric_dot_pairs(
                &q_rot, &amp_idx, &phase_idx, &sigma, &levels, &cos_lut, &sin_lut,
            )
        };
        assert!(d.abs() < 1e-7, "zero-amp dot = {d}");
        assert!(n.abs() < 1e-7, "zero-amp norm_sq = {n}");
    }

    // ============================================================
    // AVX2 block_hadamard_* parity tests (Phase 5, §4.4 / §8.1)
    //
    // The existing dispatcher-level FWHT tests already exercise the
    // AVX2 path on x86_64 hosts (the dispatcher routes block_size>=8
    // there), but these direct-call tests fail loudly if a parity
    // regression sneaks past the dispatcher (e.g. the §4.4
    // `_mm256_addsub_ps` trap re-introduced by a future "optimizer").
    // ============================================================

    #[cfg(target_arch = "x86_64")]
    fn check_avx2_fwht_round_trip(block_size: usize, seed_data: u64, seed_signs: u64) {
        if !avx2_fma_available() {
            eprintln!("skip: cpu lacks avx2+fma (fwht round-trip b={block_size})");
            return;
        }
        // Three blocks so we exercise the chunks_exact loop.
        let dim = 3 * block_size;
        let original = random_unit_vector(dim, seed_data);
        let signs = deterministic_signs(dim, seed_signs);
        let mut data = original.clone();
        // Forward via direct AVX2 entry. The dispatcher would route
        // here on x86_64 anyway; calling x86_64::* directly keeps
        // the test scope to the kernel itself.
        for (chunk, sc) in data
            .chunks_exact_mut(block_size)
            .zip(signs.chunks_exact(block_size))
        {
            unsafe { super::x86_64::fwht_forward_fused(chunk, sc) };
        }
        // Inverse and check we're back at `original` within tolerance.
        for (chunk, sc) in data
            .chunks_exact_mut(block_size)
            .zip(signs.chunks_exact(block_size))
        {
            unsafe { super::x86_64::fwht_inverse_fused(chunk, sc) };
        }
        for (i, (a, b)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                (a - b).abs() < 5e-5,
                "round-trip drift at b={block_size} idx={i}: avx={a} orig={b}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn check_avx2_fwht_matches_scalar(block_size: usize, seed: u64) {
        if !avx2_fma_available() {
            return;
        }
        let dim = 3 * block_size;
        let mut avx = random_unit_vector(dim, seed);
        let signs = deterministic_signs(dim, seed.wrapping_add(1));
        let mut scal = avx.clone();
        for (chunk, sc) in avx
            .chunks_exact_mut(block_size)
            .zip(signs.chunks_exact(block_size))
        {
            unsafe { super::x86_64::fwht_forward_fused(chunk, sc) };
        }
        scalar_block_hadamard_forward(&mut scal, &signs, block_size);
        for (i, (a, b)) in avx.iter().zip(scal.iter()).enumerate() {
            let rel = (a - b).abs() / (b.abs().max(1e-6));
            assert!(
                rel < 2e-4,
                "fwd drift at b={block_size} idx={i}: avx={a} scalar={b} rel={rel}"
            );
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_fwht_matches_scalar_b8() {
        // Smallest AVX2 block — exercises h=1, h=2, h=4 but no h>=8.
        check_avx2_fwht_matches_scalar(8, 0x01_BB_00);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_fwht_matches_scalar_b64() {
        // 64 = 2^6 → stages h ∈ {1, 2, 4, 8, 16, 32}, exercises all
        // four AVX2 stage patterns (h=1 fused, h=2 per-lane, h=4
        // cross-lane, h>=8 paired loads).
        check_avx2_fwht_matches_scalar(64, 0x02_BB_00);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_fwht_matches_scalar_b256() {
        // Production block_size at d=768.
        check_avx2_fwht_matches_scalar(256, 0x03_BB_00);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_fwht_matches_scalar_b1024() {
        // Production block_size at d=3072.
        check_avx2_fwht_matches_scalar(1024, 0x04_BB_00);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_fwht_round_trip_b256() {
        check_avx2_fwht_round_trip(256, 0x05_BB_00, 0x06_BB_00);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_fwht_round_trip_b1024() {
        check_avx2_fwht_round_trip(1024, 0x07_BB_00, 0x08_BB_00);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_fwht_preserves_norm() {
        if !avx2_fma_available() {
            return;
        }
        let dim = 1024;
        let mut data = random_unit_vector(dim, 0x09_BB_00);
        let n_before: f32 = data.iter().map(|v| v * v).sum();
        let signs = deterministic_signs(dim, 0x0A_BB_00);
        unsafe { super::x86_64::fwht_forward_fused(&mut data, &signs) };
        let n_after: f32 = data.iter().map(|v| v * v).sum();
        assert!(
            (n_before - n_after).abs() / n_before < 1e-5,
            "norm drift: before={n_before} after={n_after}"
        );
    }

    // ============================================================
    // AVX2 pair_magnitudes parity tests (Phase 6, §4.5 / §8.1)
    //
    // Elementwise output → 1e-6 absolute tolerance (no reduction
    // drift, just per-lane fp rounding). We compare against the
    // scalar reference directly.
    // ============================================================

    #[cfg(target_arch = "x86_64")]
    fn check_avx2_pair_magnitudes(p: usize, seed: u64, case: &str) {
        if !avx2_fma_available() {
            eprintln!("skip: cpu lacks avx2+fma ({case})");
            return;
        }
        let pairs = random_floats(2 * p, seed);
        let mut m_avx = vec![0.0_f32; p];
        let mut m_scalar = vec![0.0_f32; p];
        unsafe { super::x86_64::pair_magnitudes(&pairs, &mut m_avx) };
        scalar_pair_magnitudes(&pairs, &mut m_scalar);
        for (i, (a, b)) in m_avx.iter().zip(m_scalar.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "{case}: pair {i}: avx={a} scalar={b}");
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_pair_magnitudes_matches_scalar_p1024() {
        // Pure 8-pair main loop, no tail.
        check_avx2_pair_magnitudes(1024, 0x01_CC_00, "p=1024 (main loop only)");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_pair_magnitudes_matches_scalar_p384() {
        // Production size at d=768. 384 = 48 × 8 + 0, also no tail.
        check_avx2_pair_magnitudes(384, 0x02_CC_00, "p=384 (production)");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_pair_magnitudes_matches_scalar_p13() {
        // 13 = 1×8 + 1×4 + 1 scalar → exercises main, SSE tail, scalar tail.
        check_avx2_pair_magnitudes(13, 0x03_CC_00, "p=13 (all three tails)");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_pair_magnitudes_matches_scalar_p5() {
        // 5 = 1×4 + 1 scalar — no main loop iteration.
        check_avx2_pair_magnitudes(5, 0x04_CC_00, "p=5 (SSE + scalar tail)");
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx2_pair_magnitudes_known_pythagorean() {
        // Hand-checked Pythagorean inputs — no fp slop possible.
        if !avx2_fma_available() {
            return;
        }
        let pairs = vec![
            3.0, 4.0, // → 5
            5.0, 12.0, // → 13
            8.0, 6.0, // → 10
            7.0, 24.0, // → 25
            1.0, 0.0, // → 1
        ];
        let mut m = vec![0.0_f32; 5];
        unsafe { super::x86_64::pair_magnitudes(&pairs, &mut m) };
        assert!((m[0] - 5.0).abs() < 1e-6);
        assert!((m[1] - 13.0).abs() < 1e-6);
        assert!((m[2] - 10.0).abs() < 1e-6);
        assert!((m[3] - 25.0).abs() < 1e-6);
        assert!((m[4] - 1.0).abs() < 1e-6);
    }
}
