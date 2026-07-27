// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! NEON SDOT kernel for `QamSlidingEngine::raw_dot_int`. Free-function
//! signature — the engine in `super` passes its `amp_table_i8`,
//! `cos_table_i8`, `sin_table_i8` arrays explicitly so this module
//! doesn't need to reach into private engine fields.
//!
//! See the parent module docstring (and §4.3 of
//! `docs/X86_64_SIMD_PLAN.md`) for the per-iteration cycle accounting
//! and the rationale behind the `vqshrn_n_s16` round/narrow that the
//! scalar reference mirrors.

#![allow(unsafe_op_in_unsafe_fn)]

use std::arch::aarch64::*;
use std::arch::asm;

/// Inline-asm SDOT (8-byte form): `sdot Vd.2s, Vn.8b, Vm.8b`.
/// Stable Rust's `vdot_s32` is gated behind nightly's
/// `stdarch_neon_dotprod`; same trick the existing
/// `crate::codec::simd::neon_dotprod` module uses for the 16-byte
/// SDOT.
#[inline]
#[target_feature(enable = "dotprod")]
// SAFETY: requires NEON (aarch64 v8-a baseline, always present); the codec/sketch dispatcher only invokes this on aarch64.
unsafe fn sdot_8(acc: int32x2_t, a: int8x8_t, b: int8x8_t) -> int32x2_t {
    let mut out: int32x2_t = acc;
    asm!(
        "sdot {0:v}.2s, {1:v}.8b, {2:v}.8b",
        inout(vreg) out,
        in(vreg) a,
        in(vreg) b,
        options(pure, nomem, nostack),
    );
    out
}

#[target_feature(enable = "neon,dotprod")]
// SAFETY: requires NEON (aarch64 v8-a baseline, always present); the codec/sketch dispatcher only invokes this on aarch64.
// Visible to the whole `qam_sliding` module tree (its sole caller is the dispatch in
// the grandparent `qam_sliding.rs`), but no broader — the `simd/` nesting is why this is
// `pub(in …)` rather than the original `pub(super)`.
pub(in crate::codec::qam_sliding) unsafe fn raw_dot_int(
    amp_table_i8: &[i8; 32],
    cos_table_i8: &[i8; 64],
    sin_table_i8: &[i8; 64],
    num_pairs: usize,
    amp_stream: &[u8],
    phase_stream: &[u8],
    q_i8: &[i8],
) -> i64 {
    // Tables resident in NEON registers — zero L1d traffic
    // for codebook lookups inside the loop.
    let a_tbl: int8x16x2_t = vld1q_s8_x2(amp_table_i8.as_ptr());
    let c_tbl: int8x16x4_t = vld1q_s8_x4(cos_table_i8.as_ptr());
    let s_tbl: int8x16x4_t = vld1q_s8_x4(sin_table_i8.as_ptr());

    let groups = num_pairs / 8;
    let safe_groups = groups - 1;

    // Two parallel int32x2_t accumulators for ILP across SDOTs.
    let mut acc0 = vdup_n_s32(0);
    let mut acc1 = vdup_n_s32(0);

    let mut amp_ptr = amp_stream.as_ptr();
    let mut phase_ptr = phase_stream.as_ptr();
    let mut q_ptr = q_i8.as_ptr();

    for g in 0..safe_groups {
        let a_chunk = std::ptr::read_unaligned(amp_ptr as *const u64);
        let p_chunk = std::ptr::read_unaligned(phase_ptr as *const u64);

        // Extract 8 amp + 8 phase indices via fixed shifts. Apple
        // ALUs retire ~6 of these per cycle; the compiler keeps
        // them in registers and never spills to memory.
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
        let ai_vec = vld1_u8(ai_arr.as_ptr());
        let pi_vec = vld1_u8(pi_arr.as_ptr());

        let a_val = vqtbl2_s8(a_tbl, ai_vec);
        let c_val = vqtbl4_s8(c_tbl, pi_vec);
        let s_val = vqtbl4_s8(s_tbl, pi_vec);

        // `vld2_s8` deinterleaves 16 i8 bytes into (q_re, q_im) of
        // 8 lanes each.
        let q_pair = vld2_s8(q_ptr);
        let q_re = q_pair.0;
        let q_im = q_pair.1;

        // i8×i8 → i16x8 (q_re·c + q_im·s).
        let qc16 = vmull_s8(q_re, c_val);
        let qcs16 = vmlal_s8(qc16, q_im, s_val);

        // Saturating narrow with rounding: (qcs16 + 64) >> 7. With
        // q_re∈[-127,127] and c,s∈[-63,63], max |qcs16| = 16002,
        // so `>>7` lands in `[-125, 125]` — fits i8 cleanly.
        let ip = vqshrn_n_s16(qcs16, 7);

        // SDOT: 8 i8×i8 multiplications → 2 i32 lanes.
        if g & 1 == 0 {
            acc0 = sdot_8(acc0, ip, a_val);
        } else {
            acc1 = sdot_8(acc1, ip, a_val);
        }

        amp_ptr = amp_ptr.add(5);
        phase_ptr = phase_ptr.add(6);
        q_ptr = q_ptr.add(16);
    }

    // Tail: last 8 pairs via stack-buffered safe copies.
    let mut a_tail = [0u8; 8];
    let mut p_tail = [0u8; 8];
    std::ptr::copy_nonoverlapping(amp_ptr, a_tail.as_mut_ptr(), 5);
    std::ptr::copy_nonoverlapping(phase_ptr, p_tail.as_mut_ptr(), 6);
    let a_chunk = u64::from_le_bytes(a_tail);
    let p_chunk = u64::from_le_bytes(p_tail);
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
    let ai_vec = vld1_u8(ai_arr.as_ptr());
    let pi_vec = vld1_u8(pi_arr.as_ptr());
    let a_val = vqtbl2_s8(a_tbl, ai_vec);
    let c_val = vqtbl4_s8(c_tbl, pi_vec);
    let s_val = vqtbl4_s8(s_tbl, pi_vec);
    let q_pair = vld2_s8(q_ptr);
    let qc16 = vmull_s8(q_pair.0, c_val);
    let qcs16 = vmlal_s8(qc16, q_pair.1, s_val);
    let ip = vqshrn_n_s16(qcs16, 7);
    acc0 = sdot_8(acc0, ip, a_val);

    // Reduce two int32x2_t accumulators down to a scalar.
    let acc = vadd_s32(acc0, acc1);
    let lane0 = vget_lane_s32(acc, 0) as i64;
    let lane1 = vget_lane_s32(acc, 1) as i64;
    lane0 + lane1
}
