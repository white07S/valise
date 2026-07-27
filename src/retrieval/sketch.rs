// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Sign-sketch candidate generation for vector search.
//!
//! Replaces the old CSR vote index. Each active vector contributes a
//! 1-bit-per-dimension sign sketch of its rotated reconstruction —
//! derived for free from the QAM phase codes at file-open (see
//! `crate::codec::qam_sliding::QamSlidingEngine::sign_sketch`), so
//! nothing extra is persisted. At query time we popcount-Hamming the
//! query's sign sketch against every active vector's (full coverage,
//! O(N) but cache-sequential SIMD) and counting-sort the closest
//! `channel_k` as candidates. The QAM-sliding rerank then scores those
//! candidates exactly (`sketch_then_rerank_impl`'s second half).
//!
//! See `bench/results/sweep/` for the design rationale: full-resolution
//! sign bits are the minimum faithful coverage (LSH / PCA / shorter
//! sketches all lose neighborhoods), and the scan is cheap-linear — the
//! candidate-gen work that the dense popcount keeps small lets the
//! rerank stay tiny.
//!
//! ## Per-arch dispatch shape (Phase 1)
//!
//! Per §4.2 of `docs/X86_64_SIMD_PLAN.md`, the AVX2 dispatcher must wrap
//! the WHOLE scan loop, not the per-vector hamming — `#[target_feature]`
//! functions are not inlinable into non-target-feature callers on stable
//! Rust, so a naive per-vector dispatch would pay a non-inlinable call
//! boundary ~N times per query. Phase 1 sets up the structural shape:
//!
//! - `hamming_scalar` — the always-available scalar reference.
//! - `hamming` — per-vector dispatcher (aarch64 → NEON via the
//!   `simd::aarch64` module; otherwise → `hamming_scalar`). aarch64's
//!   NEON is baseline-ISA, so the per-call shape is fine there.
//! - `scan_candidates_scalar` — the legacy loop body. Uses `hamming()`,
//!   so on aarch64 it still gets the NEON win.
//! - `scan_candidates_avx2` — Phase 1 stub that delegates to
//!   `_scalar`. Phase 2 rewrites this body to inline AVX2 hamming
//!   under the same `#[target_feature(enable = "avx2")]` context.
//! - `scan_candidates` — public entry; on x86_64 with runtime AVX2,
//!   routes through `_avx2`. Everywhere else: straight to `_scalar`.

use crate::format::VectorId;

mod simd;

/// Bench-only re-exposure of the AVX2 hamming kernels so the
/// criterion bench in `benches/hamming.rs` can time them in isolation
/// (per §4.2 the production path reaches them via
/// `scan_candidates_avx2`, never via the per-vector dispatcher).
///
/// Thin wrapper fns (not `pub use`) because the underlying kernels
/// are `pub(crate)` — same pattern as the codec's `simd_bench`.
#[cfg(all(target_arch = "x86_64", any(test, feature = "bench")))]
pub(crate) mod hamming_x86_64_bench {
    /// SAFETY: caller must have verified `avx2` and `popcnt` CPU
    /// support. Defers to the kernel's own target_feature gate.
    #[inline(always)]
    pub unsafe fn hamming_popcnt(a: &[u64], b: &[u64]) -> u32 {
        unsafe { super::simd::x86_64::hamming_popcnt(a, b) }
    }

    /// SAFETY: caller must have verified `avx2` and `popcnt` CPU
    /// support. Defers to the kernel's own target_feature gate.
    #[inline(always)]
    pub unsafe fn hamming_shuffle(a: &[u64], b: &[u64]) -> u32 {
        unsafe { super::simd::x86_64::hamming_shuffle(a, b) }
    }
}

/// Pack the 1-bit-per-dim sign sketch of a rotated query into u64 words.
/// Matches [`QamSlidingEngine::sign_sketch`]'s DB-side packing (bit set
/// when the rotated coordinate is ≥ 0).
pub(crate) fn pack_query_sketch(rot: &[f32]) -> Vec<u64> {
    let mut o = vec![0u64; rot.len().div_ceil(64)];
    for (i, &x) in rot.iter().enumerate() {
        if x >= 0.0 {
            o[i >> 6] |= 1u64 << (i & 63);
        }
    }
    o
}

#[inline(always)]
pub(crate) fn hamming_scalar(a: &[u64], b: &[u64]) -> u32 {
    let mut h = 0u32;
    for w in 0..a.len() {
        h += (a[w] ^ b[w]).count_ones();
    }
    h
}

#[inline]
pub(crate) fn hamming(a: &[u64], b: &[u64]) -> u32 {
    #[cfg(target_arch = "aarch64")]
    unsafe {
        return simd::aarch64::hamming(a, b);
    }
    #[allow(unreachable_code)]
    hamming_scalar(a, b)
}

/// Hamming-scan the query sketch against every active vector's sketch and
/// return the `ck` closest as `(vid, hamming)` candidates (unsorted; the
/// rerank only consumes the vids). `rows[i].0` is the vid for sketch row
/// `i`; `dim` bounds the histogram (max Hamming = dim). `hbuf` is reused
/// across queries to avoid per-query allocation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn scan_candidates(
    sketches: &[u64],
    words: usize,
    rows: &[(u32, usize)],
    q_sketch: &[u64],
    ck: usize,
    dim: usize,
    parallel: bool,
    hbuf: &mut Vec<u16>,
) -> Vec<(VectorId, i32)> {
    #[cfg(all(
        target_arch = "x86_64",
        target_feature = "avx2",
        target_feature = "popcnt"
    ))]
    return unsafe {
        // Static dispatch — both target_features guaranteed at
        // compile time. `scan_candidates_avx2` then inlines into
        // this caller, eliminating §4.2's per-vector call boundary
        // concern even at the dispatcher level.
        scan_candidates_avx2(sketches, words, rows, q_sketch, ck, dim, parallel, hbuf)
    };
    #[cfg(all(
        target_arch = "x86_64",
        not(all(target_feature = "avx2", target_feature = "popcnt"))
    ))]
    {
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt") {
            return unsafe {
                scan_candidates_avx2(sketches, words, rows, q_sketch, ck, dim, parallel, hbuf)
            };
        }
    }
    // Dead by construction when avx2+popcnt are static (the x86-64-v3
    // baseline in .cargo/config.toml): the first arm above returns. That is
    // the point of the cfg ladder, not an oversight.
    #[allow(unreachable_code)]
    scan_candidates_scalar(sketches, words, rows, q_sketch, ck, dim, parallel, hbuf)
}

/// Words threshold for choosing between the two AVX2 hamming variants.
/// Below this, the PSHUFB shuffle-popcount wins because its 32-byte
/// vectorized chunk pays off even with 1–2 chunks; the POPCNT
/// alternative has a stronger dependent chain at small widths.
/// At/above, scalar 64-bit POPCNT in a tight loop wins (more
/// instructions but no shuffle setup, and the dependent chain
/// dissolves into the OOO machine).
///
/// Crossover measured on the lab i7-8550U at Phase 2 close-out
/// (`benches/baseline/scalar_i7_8550u.json`):
///
/// | words | shuffle | popcnt | winner  |
/// |-------|--------:|-------:|---------|
/// |     8 |    5.38 |   6.16 | shuffle |
/// |    12 |    6.66 |   7.35 | shuffle |
/// |    16 |    7.99 |   8.34 | shuffle |
/// |    32 |   13.63 |  13.01 | popcnt  |
/// |    64 |   23.93 |  21.64 | popcnt  |
///
/// 20 sits between the 16 and 32 measurement points so the production
/// widths (12 at dim=768; 48 at dim=3072) both land on the right
/// variant.
#[cfg(target_arch = "x86_64")]
const HAMMING_POPCNT_WORDS_THRESHOLD: usize = 20;

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn scan_candidates_avx2(
    sketches: &[u64],
    words: usize,
    rows: &[(u32, usize)],
    q_sketch: &[u64],
    ck: usize,
    dim: usize,
    parallel: bool,
    hbuf: &mut Vec<u16>,
) -> Vec<(VectorId, i32)> {
    // §4.2 structural rule: the AVX2 boundary wraps the WHOLE scan
    // loop, not the per-vector hamming. We dispatch to a const-
    // specialized inner fn so the chosen hamming variant inlines
    // inside the per-vector inner loop and the dispatch branch lives
    // only at scan entry (once per query, not once per vector).
    // SAFETY: this fn is itself target_feature-gated on avx2+popcnt;
    // the call into the const-specialized inner inherits the same
    // feature set so the unsafe boundary is closed at scan entry.
    // `USE_SHUFFLE = true` selects the PSHUFB nibble-LUT path
    // (faster at small widths); `false` selects the scalar-POPCNT
    // path (faster at words ≥ 20 — see crossover table above).
    if words < HAMMING_POPCNT_WORDS_THRESHOLD {
        unsafe {
            scan_candidates_avx2_inner::<true>(
                sketches, words, rows, q_sketch, ck, dim, parallel, hbuf,
            )
        }
    } else {
        unsafe {
            scan_candidates_avx2_inner::<false>(
                sketches, words, rows, q_sketch, ck, dim, parallel, hbuf,
            )
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,popcnt")]
#[allow(clippy::too_many_arguments)]
unsafe fn scan_candidates_avx2_inner<const USE_SHUFFLE: bool>(
    sketches: &[u64],
    words: usize,
    rows: &[(u32, usize)],
    q_sketch: &[u64],
    ck: usize,
    dim: usize,
    parallel: bool,
    hbuf: &mut Vec<u16>,
) -> Vec<(VectorId, i32)> {
    use rayon::prelude::*;

    // SAFETY: caller holds AVX2 + POPCNT target_feature gate (this
    // fn is target_feature-enabled). The const generic monomorphizes
    // one specialization per hamming variant so the inner branch is
    // dead at codegen time and the chosen kernel inlines into the
    // loop body. We use plain `#[inline]` because stable Rust does
    // not allow `#[inline(always)]` on `#[target_feature]` fns
    // (gated behind `target_feature_11`); the regular hint is enough
    // here because the caller is itself target_feature-enabled and
    // the inner kernel is small.
    #[inline]
    #[target_feature(enable = "avx2,popcnt")]
    unsafe fn ham_inner<const SHUFFLE: bool>(q: &[u64], v: &[u64]) -> u32 {
        if SHUFFLE {
            unsafe { simd::x86_64::hamming_shuffle(q, v) }
        } else {
            unsafe { simd::x86_64::hamming_popcnt(q, v) }
        }
    }

    let n = rows.len();
    if n == 0 {
        return Vec::new();
    }
    hbuf.clear();
    hbuf.resize(n, 0);

    let hist: Vec<u32> = if parallel {
        // The closure inherits this fn's target_feature context, so
        // the SAFETY of calling `ham_inner` (target-feature-gated)
        // from inside is satisfied. Each closure invocation runs the
        // per-chunk loop, so the call boundary into the closure is
        // amortized over 4096 hamming calls (and the hamming variant
        // is inlined inside the const-specialized inner loop).
        hbuf.par_chunks_mut(4096)
            .enumerate()
            .map(|(c, chunk)| {
                let base = c * 4096;
                let mut local = vec![0u32; dim + 2];
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let i = base + j;
                    let h = unsafe {
                        ham_inner::<USE_SHUFFLE>(q_sketch, &sketches[i * words..(i + 1) * words])
                    };
                    *slot = h as u16;
                    local[h as usize] += 1;
                }
                local
            })
            .reduce(
                || vec![0u32; dim + 2],
                |mut a, b| {
                    for d in 0..a.len() {
                        a[d] += b[d];
                    }
                    a
                },
            )
    } else {
        let mut hh = vec![0u32; dim + 2];
        for (i, slot) in hbuf.iter_mut().enumerate() {
            let h = unsafe {
                ham_inner::<USE_SHUFFLE>(q_sketch, &sketches[i * words..(i + 1) * words])
            };
            *slot = h as u16;
            hh[h as usize] += 1;
        }
        hh
    };

    let mut acc = 0u32;
    let mut thr = dim + 1;
    for (d, &c) in hist.iter().enumerate() {
        if acc + c >= ck as u32 {
            thr = d;
            break;
        }
        acc += c;
    }
    let mut rem = (ck as u32).saturating_sub(acc);

    let mut out: Vec<(VectorId, i32)> = Vec::with_capacity(ck + 64);
    for (i, &h) in hbuf.iter().enumerate() {
        let h = h as usize;
        let take = if h < thr {
            true
        } else if h == thr && rem > 0 {
            rem -= 1;
            true
        } else {
            false
        };
        if take {
            out.push((VectorId(rows[i].0 as u64), h as i32));
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn scan_candidates_scalar(
    sketches: &[u64],
    words: usize,
    rows: &[(u32, usize)],
    q_sketch: &[u64],
    ck: usize,
    dim: usize,
    parallel: bool,
    hbuf: &mut Vec<u16>,
) -> Vec<(VectorId, i32)> {
    use rayon::prelude::*;
    let n = rows.len();
    if n == 0 {
        return Vec::new();
    }
    hbuf.clear();
    hbuf.resize(n, 0);

    // Fused distance + chunk-local histogram: one pass, no global atomics,
    // no second pass over hbuf to build the histogram.
    let hist: Vec<u32> = if parallel {
        hbuf.par_chunks_mut(4096)
            .enumerate()
            .map(|(c, chunk)| {
                let base = c * 4096;
                let mut local = vec![0u32; dim + 2];
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let i = base + j;
                    let h = hamming(q_sketch, &sketches[i * words..(i + 1) * words]);
                    *slot = h as u16;
                    local[h as usize] += 1;
                }
                local
            })
            .reduce(
                || vec![0u32; dim + 2],
                |mut a, b| {
                    for d in 0..a.len() {
                        a[d] += b[d];
                    }
                    a
                },
            )
    } else {
        let mut hh = vec![0u32; dim + 2];
        for (i, slot) in hbuf.iter_mut().enumerate() {
            let h = hamming(q_sketch, &sketches[i * words..(i + 1) * words]);
            *slot = h as u16;
            hh[h as usize] += 1;
        }
        hh
    };

    // Counting-sort threshold: the `ck` smallest Hamming, no full sort.
    let mut acc = 0u32;
    let mut thr = dim + 1;
    for (d, &c) in hist.iter().enumerate() {
        if acc + c >= ck as u32 {
            thr = d;
            break;
        }
        acc += c;
    }
    let mut rem = (ck as u32).saturating_sub(acc);

    let mut out: Vec<(VectorId, i32)> = Vec::with_capacity(ck + 64);
    for (i, &h) in hbuf.iter().enumerate() {
        let h = h as usize;
        let take = if h < thr {
            true
        } else if h == thr && rem > 0 {
            rem -= 1;
            true
        } else {
            false
        };
        if take {
            out.push((VectorId(rows[i].0 as u64), h as i32));
        }
    }
    out
}
