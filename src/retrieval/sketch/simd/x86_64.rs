//! AVX2 popcount-Hamming kernels for the sign-sketch scan
//! (`docs/X86_64_SIMD_PLAN.md` §4.2).
//!
//! Two variants:
//!
//! - [`hamming_popcnt`] — scalar 64-bit `_mm_popcnt_u64` loop wrapped
//!   in an AVX2 target-feature context. Wins at small sketch widths
//!   (dim ≤ 512) where the AVX2 setup + reduction doesn't amortize.
//! - [`hamming_shuffle`] — AVX2 PSHUFB nibble-LUT popcount with
//!   per-chunk SAD (Muła–Kurz–Lemire 2017). Wins at larger widths
//!   (dim ≥ 768 / words ≥ 12).
//!
//! **Critical**: per §4.2 the AVX2 boundary on x86_64 wraps the WHOLE
//! scan loop (not the per-vector hamming) because
//! `#[target_feature(enable = "avx2")]` functions cannot be inlined
//! into non-target-feature callers on stable Rust. Both functions
//! below are `#[inline(always)]` and intended to be called from
//! inside `super::scan_candidates_avx2`, which carries the same
//! target_feature enable. Calling them from anywhere else negates
//! the inlining win.

#![allow(unsafe_op_in_unsafe_fn)]

use core::arch::x86_64::*;

#[target_feature(enable = "avx2,popcnt")]
#[inline]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
pub(crate) unsafe fn hamming_popcnt(a: &[u64], b: &[u64]) -> u32 {
    // POPCNT on Intel since Nehalem: 3-cycle latency, 1-per-cycle
    // throughput. For ≤ 8 words the dependent-chain length is short
    // enough that ILP across the loop body keeps a single popcnt port
    // busy without enough work to amortize a vector setup.
    debug_assert_eq!(a.len(), b.len());
    let mut h: u64 = 0;
    for w in 0..a.len() {
        // `u64::count_ones()` lowers to `popcnt` under the
        // target_feature gate; using the intrinsic directly removes
        // any doubt about codegen.
        h += _popcnt64((a[w] ^ b[w]) as i64) as u64;
    }
    h as u32
}

#[target_feature(enable = "avx2,popcnt")]
#[inline]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
pub(crate) unsafe fn hamming_shuffle(a: &[u64], b: &[u64]) -> u32 {
    // Muła–Kurz–Lemire 2017: PSHUFB nibble-LUT popcount, per-chunk
    // SAD into u64 lanes. For words ∈ {8, 12, 16, 32, 64} the byte
    // count is a multiple of 32 so the tail loop never executes; we
    // still emit it for general words counts.
    debug_assert_eq!(a.len(), b.len());
    let nb = a.len() * 8;
    let pa = a.as_ptr() as *const u8;
    let pb = b.as_ptr() as *const u8;

    // Per-nibble popcount LUT, replicated into both 128-bit lanes
    // (PSHUFB is per-128-bit-lane on AVX2).
    let nibble_lut = _mm256_setr_epi8(
        0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3,
        3, 4,
    );
    let mask_lo = _mm256_set1_epi8(0x0F);
    let mut acc = _mm256_setzero_si256();

    let mut i = 0usize;
    while i + 32 <= nb {
        let av = _mm256_loadu_si256(pa.add(i) as *const __m256i);
        let bv = _mm256_loadu_si256(pb.add(i) as *const __m256i);
        let x = _mm256_xor_si256(av, bv);
        let lo = _mm256_and_si256(x, mask_lo);
        // PSRLI on epi16 (no epi8 variant) — bits 4..7 of the next byte
        // leak into bits 0..3 of the current byte after the shift, but
        // mask_lo clamps them off.
        let hi = _mm256_and_si256(_mm256_srli_epi16(x, 4), mask_lo);
        let pop_lo = _mm256_shuffle_epi8(nibble_lut, lo);
        let pop_hi = _mm256_shuffle_epi8(nibble_lut, hi);
        // byte-wise popcount, each lane 0..8.
        let bytes = _mm256_add_epi8(pop_lo, pop_hi);
        // SAFETY: per-chunk SAD prevents byte overflow. Each i8 lane
        // here is 0..8; accumulating across 32 chunks would saturate
        // at 256. Keeping SAD inside the loop body widens to u64
        // immediately, and u64 won't overflow for any realistic
        // sketch width (max sum is ~64 · 2^16 = 4 M bits).
        let sums = _mm256_sad_epu8(bytes, _mm256_setzero_si256());
        acc = _mm256_add_epi64(acc, sums);
        i += 32;
    }

    // Reduce 4 × u64 lanes to scalar.
    let mut lanes = [0u64; 4];
    _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, acc);
    let mut h = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    // Tail bytes (words count not a multiple of 4 — never the case
    // for the canonical sketch widths but we keep the loop for
    // correctness on general inputs).
    while i < nb {
        h += _popcnt64((*pa.add(i) ^ *pb.add(i)) as i64) as u64;
        i += 1;
    }
    h as u32
}
