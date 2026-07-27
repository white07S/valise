//! AVX2 i8 SDOT-equivalent kernel for `QamSlidingEngine::raw_dot_int`
//! (`docs/X86_64_SIMD_PLAN.md` §4.3). The hardest port of the AVX2
//! rollout because:
//!
//! 1. **No SDOT.** `VPDPBUSD` (AVX-VNNI) doesn't exist on Kaby Lake R
//!    or older. We replace `vmull_s8 + vmlal_s8 + sdot` with a chain
//!    of `_mm_madd_epi16` (i16×i16 → i32 with adjacent-pair sum) +
//!    explicit i32 accumulate.
//!
//! 2. **No fast 64-byte register-resident table lookup.** NEON's
//!    `vqtbl4_s8` indexes a 64-byte table in one cycle from a single
//!    6-bit index per lane. AVX2's `_mm256_shuffle_epi8` is per-128-
//!    bit-lane with 4-bit indices — emulating TBL4 costs more than
//!    it's worth. We accept scalar table gathers; the 32+64+64 = 160
//!    B of LUTs stays L1d-resident.
//!
//! 3. **Bit-exact `vqshrn_n_s16(x, 7)` reproduction.** The integer
//!    rerank's output feeds a top-k comparator; a single LSB flip
//!    can reorder near-tied candidates and change recall. The §4.3
//!    canonical sequence `add(1<<6) → srai 7 → packs` matches NEON's
//!    rounding-saturating narrow exactly under the codec's
//!    documented `|qcs16| ≤ 16002` window. Anyone modifying the
//!    arithmetic here must re-verify against the scalar reference's
//!    bit pattern, not just "close enough" floating-point tolerance.
//!
//! Per-iteration cost (8 pairs, mirrors NEON's iteration shape):
//! ~16 GPR shifts/masks for bit-unpack + ~14 AVX2 µops for the
//! widen-multiply-narrow-widen-multiply-accumulate chain. Expected
//! ~2-3× slower than NEON SDOT (no SDOT-equivalent, no register-
//! resident TBL). Phase 4 baseline target per §9.2: ≥ scalar.

#![allow(dead_code)]
#![allow(unsafe_op_in_unsafe_fn)]
#![allow(clippy::too_many_arguments)]

use core::arch::x86_64::*;

#[target_feature(enable = "avx2")]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
pub(crate) unsafe fn raw_dot_int(
    amp_table_i8: &[i8; 32],
    cos_table_i8: &[i8; 64],
    sin_table_i8: &[i8; 64],
    num_pairs: usize,
    amp_stream: &[u8],
    phase_stream: &[u8],
    q_i8: &[i8],
) -> i64 {
    debug_assert!(num_pairs.is_multiple_of(8) && num_pairs >= 8);

    let groups = num_pairs / 8;
    let safe_groups = groups - 1;

    // Two parallel i32x8 accumulators for ILP. Accumulator overflow
    // bound: per-pair |q·c + q·s| ≤ 2·127·63 = 16002 (the documented
    // codec window); after rounding-saturating narrow to i8 each
    // intermediate is in [-126, 125]; amp ∈ [0, 127]; product ∈
    // [-126·127, 125·127] ⊂ [-16002, 15875]. Over P = 1536 pairs,
    // max |acc| ≤ 1536 · 16002 = 24.6 M, well below i32::MAX = 2.1 B.
    // Safe to use `_mm256_mullo_epi32` + `_mm256_add_epi32` without
    // overflow checks.
    let mut acc0 = _mm256_setzero_si256();
    let mut acc1 = _mm256_setzero_si256();

    let mut amp_ptr = amp_stream.as_ptr();
    let mut phase_ptr = phase_stream.as_ptr();
    let mut q_ptr = q_i8.as_ptr();

    // PSHUFB mask: pull even bytes (q_re) to slots 0..7 and odd
    // bytes (q_im) to slots 8..15.
    let q_split_mask = _mm_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, // even-index bytes → q_re
        1, 3, 5, 7, 9, 11, 13, 15, // odd-index bytes  → q_im
    );
    // `vqshrn_n_s16(x, 7)` bias.
    let round_bias = _mm_set1_epi16(1 << 6);

    for g in 0..safe_groups {
        let prod = process_group(
            amp_table_i8,
            cos_table_i8,
            sin_table_i8,
            amp_ptr,
            phase_ptr,
            q_ptr,
            q_split_mask,
            round_bias,
        );
        if g & 1 == 0 {
            acc0 = _mm256_add_epi32(acc0, prod);
        } else {
            acc1 = _mm256_add_epi32(acc1, prod);
        }
        amp_ptr = amp_ptr.add(5);
        phase_ptr = phase_ptr.add(6);
        q_ptr = q_ptr.add(16);
    }

    // Tail: last 8 pairs via stack-buffered safe copies. Mirrors
    // NEON exactly — the final 5-byte / 6-byte chunks would
    // overflow if read via unaligned u64.
    let mut a_tail = [0u8; 8];
    let mut p_tail = [0u8; 8];
    core::ptr::copy_nonoverlapping(amp_ptr, a_tail.as_mut_ptr(), 5);
    core::ptr::copy_nonoverlapping(phase_ptr, p_tail.as_mut_ptr(), 6);
    let prod = process_group_from_chunks(
        amp_table_i8,
        cos_table_i8,
        sin_table_i8,
        u64::from_le_bytes(a_tail),
        u64::from_le_bytes(p_tail),
        q_ptr,
        q_split_mask,
        round_bias,
    );
    acc0 = _mm256_add_epi32(acc0, prod);

    // Reduce: acc0 + acc1, then horizontal sum across 8 i32 lanes.
    let acc = _mm256_add_epi32(acc0, acc1);
    let lo = _mm256_castsi256_si128(acc);
    let hi = _mm256_extracti128_si256(acc, 1);
    let s4 = _mm_add_epi32(lo, hi);
    let s2 = _mm_add_epi32(s4, _mm_shuffle_epi32(s4, 0b00_00_11_10));
    let s1 = _mm_add_epi32(s2, _mm_shuffle_epi32(s2, 0b00_00_00_01));
    _mm_cvtsi128_si32(s1) as i64
}

/// Process one 8-pair group reading from packed `amp_ptr` and
/// `phase_ptr` via unaligned `u64` loads (caller must guarantee
/// 8 bytes are readable past each pointer).
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
unsafe fn process_group(
    amp_table_i8: &[i8; 32],
    cos_table_i8: &[i8; 64],
    sin_table_i8: &[i8; 64],
    amp_ptr: *const u8,
    phase_ptr: *const u8,
    q_ptr: *const i8,
    q_split_mask: __m128i,
    round_bias: __m128i,
) -> __m256i {
    let a_chunk = core::ptr::read_unaligned(amp_ptr as *const u64);
    let p_chunk = core::ptr::read_unaligned(phase_ptr as *const u64);
    process_group_from_chunks(
        amp_table_i8,
        cos_table_i8,
        sin_table_i8,
        a_chunk,
        p_chunk,
        q_ptr,
        q_split_mask,
        round_bias,
    )
}

/// Inner kernel: takes already-loaded amp/phase chunks. Identical
/// to NEON's per-iteration body up to instruction mapping.
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
unsafe fn process_group_from_chunks(
    amp_table_i8: &[i8; 32],
    cos_table_i8: &[i8; 64],
    sin_table_i8: &[i8; 64],
    a_chunk: u64,
    p_chunk: u64,
    q_ptr: *const i8,
    q_split_mask: __m128i,
    round_bias: __m128i,
) -> __m256i {
    // ---- Bit-unpack 8 amp + 8 phase indices via GPR shifts ----
    let ai_arr: [u8; 8] = [
        ((a_chunk) & 0x1F) as u8,
        ((a_chunk >> 5) & 0x1F) as u8,
        ((a_chunk >> 10) & 0x1F) as u8,
        ((a_chunk >> 15) & 0x1F) as u8,
        ((a_chunk >> 20) & 0x1F) as u8,
        ((a_chunk >> 25) & 0x1F) as u8,
        ((a_chunk >> 30) & 0x1F) as u8,
        ((a_chunk >> 35) & 0x1F) as u8,
    ];
    let pi_arr: [u8; 8] = [
        ((p_chunk) & 0x3F) as u8,
        ((p_chunk >> 6) & 0x3F) as u8,
        ((p_chunk >> 12) & 0x3F) as u8,
        ((p_chunk >> 18) & 0x3F) as u8,
        ((p_chunk >> 24) & 0x3F) as u8,
        ((p_chunk >> 30) & 0x3F) as u8,
        ((p_chunk >> 36) & 0x3F) as u8,
        ((p_chunk >> 42) & 0x3F) as u8,
    ];

    // ---- Scalar table gather. 160 B of LUTs stays L1d-resident. ----
    let mut a_vals = [0i8; 8];
    let mut c_vals = [0i8; 8];
    let mut s_vals = [0i8; 8];
    for i in 0..8 {
        a_vals[i] = *amp_table_i8.get_unchecked(ai_arr[i] as usize);
        c_vals[i] = *cos_table_i8.get_unchecked(pi_arr[i] as usize);
        s_vals[i] = *sin_table_i8.get_unchecked(pi_arr[i] as usize);
    }

    // ---- Load (re, im) interleaved q bytes, deinterleave via PSHUFB ----
    let q16 = _mm_loadu_si128(q_ptr as *const __m128i);
    let q_split = _mm_shuffle_epi8(q16, q_split_mask);
    // Low 8 bytes: q_re. High 8 bytes: q_im. Widen each to 8 i16 lanes.
    let q_re_i16 = _mm_cvtepi8_epi16(q_split);
    let q_im_i16 = _mm_cvtepi8_epi16(_mm_srli_si128(q_split, 8));

    // ---- Load c, s tables; widen to i16 BEFORE interleave (§4.3) ----
    let c_i8 = _mm_loadl_epi64(c_vals.as_ptr() as *const __m128i);
    let s_i8 = _mm_loadl_epi64(s_vals.as_ptr() as *const __m128i);
    let c_i16 = _mm_cvtepi8_epi16(c_i8);
    let s_i16 = _mm_cvtepi8_epi16(s_i8);

    // ---- `q·c + q·s` via _mm_madd_epi16 ----
    // PMADDWD does (a0·b0 + a1·b1) over adjacent i16 pairs → 4 i32
    // lanes per call. Arrange inputs so adjacent pairs hold
    // (q_re[k], q_im[k]) and (c[k], s[k]), giving us inner[k] =
    // q_re[k]·c[k] + q_im[k]·s[k] in one instruction per 4 pairs.
    let q_pairs_lo = _mm_unpacklo_epi16(q_re_i16, q_im_i16);
    let q_pairs_hi = _mm_unpackhi_epi16(q_re_i16, q_im_i16);
    let cs_pairs_lo = _mm_unpacklo_epi16(c_i16, s_i16);
    let cs_pairs_hi = _mm_unpackhi_epi16(c_i16, s_i16);
    let inner_lo = _mm_madd_epi16(q_pairs_lo, cs_pairs_lo);
    let inner_hi = _mm_madd_epi16(q_pairs_hi, cs_pairs_hi);

    // ---- Pack i32 → i16 with signed saturation ----
    // Per the codec's `|qcs16| ≤ 16002` documented window, no
    // saturation occurs; the pack is purely a width change.
    let qcs16 = _mm_packs_epi32(inner_lo, inner_hi);
    debug_assert_bounds_qcs16(qcs16);

    // ---- Reproduce NEON's vqshrn_n_s16(x, 7) bit-exactly ----
    // The §4.3 canonical sequence: add(1<<6) → srai 7 → packs.
    // Within the codec's |qcs16+64| ≤ 16066 ⊂ i16 window the add
    // can't overflow and `_mm_packs_epi16` saturates at the i8
    // boundary exactly like vqshrn's narrowing step.
    let biased = _mm_add_epi16(qcs16, round_bias);
    let shifted = _mm_srai_epi16(biased, 7);
    let ip_i8 = _mm_packs_epi16(shifted, shifted); // 8 i8s in low 64 bits

    // ---- Widen ip and a to 8 i32 each, multiply, accumulate ----
    let a_i8 = _mm_loadl_epi64(a_vals.as_ptr() as *const __m128i);
    let ip_i32 = _mm256_cvtepi8_epi32(ip_i8);
    let a_i32 = _mm256_cvtepi8_epi32(a_i8);
    _mm256_mullo_epi32(ip_i32, a_i32)
}

#[cfg(debug_assertions)]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
unsafe fn debug_assert_bounds_qcs16(qcs16: __m128i) {
    // The codec invariant |qcs16| ≤ 16002 is what lets the
    // post-PACKS arithmetic skip overflow handling. Validate in
    // debug builds; release omits the check.
    let mut lanes = [0i16; 8];
    _mm_storeu_si128(lanes.as_mut_ptr() as *mut __m128i, qcs16);
    for (i, &v) in lanes.iter().enumerate() {
        debug_assert!(
            v.unsigned_abs() <= 16002,
            "raw_dot_int_x86_64: |qcs16[{i}]| = {} exceeds the documented 16002 bound \
             (q_re∈[-127,127], c,s∈[-63,63]); the round-narrow path's overflow guarantees \
             no longer hold",
            v.unsigned_abs()
        );
    }
}

#[cfg(not(debug_assertions))]
#[target_feature(enable = "avx2")]
#[inline]
// SAFETY: requires the enclosing #[target_feature] (AVX2/FMA/POPCNT); upheld by static target-cpu=x86-64-v3 or the dispatcher's runtime is_x86_feature_detected! check.
unsafe fn debug_assert_bounds_qcs16(_qcs16: __m128i) {}
