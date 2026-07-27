//! NEON popcount-Hamming kernel for the sign-sketch scan. Lifted
//! verbatim from `super::sketch` during the Phase 1 scaffolding split
//! (see `docs/X86_64_SIMD_PLAN.md` §5.3); behavior is bit-identical
//! to the previous in-file NEON arm.

#![allow(unsafe_op_in_unsafe_fn)]

/// Popcount-Hamming distance between two sign-sketches. `a.len() == b.len()`
/// is the per-vector word count.
#[inline]
// SAFETY: requires NEON (aarch64 v8-a baseline, always present); the codec/sketch dispatcher only invokes this on aarch64.
// Visible to the whole `sketch` module tree (its sole caller is the `hamming` dispatcher
// in the grandparent `sketch.rs`), but no broader — the `simd/` nesting is why this is
// `pub(in …)` rather than the original `pub(super)`.
pub(in crate::retrieval::sketch) unsafe fn hamming(a: &[u64], b: &[u64]) -> u32 {
    use std::arch::aarch64::*;
    let nb = a.len() * 8;
    let (pa, pb) = (a.as_ptr() as *const u8, b.as_ptr() as *const u8);
    // NEON CNT: XOR + vcntq_u8 over the whole sketch, accumulate
    // per-byte (≤ 48 chunks × 8 fits before overflow at dim ≤ 24576),
    // one vaddlvq reduce. No per-u64 GPR↔SIMD round-trips.
    let mut s = vdupq_n_u16(0);
    let mut i = 0;
    while i + 16 <= nb {
        let x = veorq_u8(vld1q_u8(pa.add(i)), vld1q_u8(pb.add(i)));
        s = vaddq_u16(s, vpaddlq_u8(vcntq_u8(x)));
        i += 16;
    }
    let mut h = vaddvq_u16(s) as u32;
    while i < nb {
        h += (*pa.add(i) ^ *pb.add(i)).count_ones();
        i += 1;
    }
    h
}
