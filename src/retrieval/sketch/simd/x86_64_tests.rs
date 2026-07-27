// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Bit-exact parity tests for the AVX2 hamming variants (Phase 2,
//! §4.2 / §8.1 of `docs/X86_64_SIMD_PLAN.md`).
//!
//! Both variants are integer kernels — zero tolerance vs the scalar
//! reference. Inputs use a deterministic `SplitMix64` so the test
//! sequence is byte-identical across runs and arches.

use super::super::hamming_scalar;
use super::x86_64 as hamming_x86_64;

struct SplitMix64(u64);
impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

fn pair(words: usize, seed: u64) -> (Vec<u64>, Vec<u64>) {
    let mut prng = SplitMix64::new(seed);
    let a: Vec<u64> = (0..words).map(|_| prng.next_u64()).collect();
    let b: Vec<u64> = (0..words).map(|_| prng.next_u64()).collect();
    (a, b)
}

fn avx2_available() -> bool {
    std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt")
}

#[test]
fn popcnt_matches_scalar_all_widths() {
    if !avx2_available() {
        eprintln!("skip: cpu lacks avx2+popcnt");
        return;
    }
    for &words in &[1_usize, 4, 8, 12, 16, 32, 64, 96] {
        for seed in [0xCAFE_BABE, 0x0000_0000, 0xFFFF_FFFF, 0xA5A5_A5A5] {
            let (a, b) = pair(words, seed);
            let want = hamming_scalar(&a, &b);
            let got = unsafe { hamming_x86_64::hamming_popcnt(&a, &b) };
            assert_eq!(got, want, "popcnt mismatch words={words} seed={seed:#x}");
        }
    }
}

#[test]
fn shuffle_matches_scalar_all_widths() {
    if !avx2_available() {
        eprintln!("skip: cpu lacks avx2+popcnt");
        return;
    }
    for &words in &[1_usize, 4, 8, 12, 16, 32, 64, 96] {
        for seed in [0xCAFE_BABE, 0x0000_0000, 0xFFFF_FFFF, 0xA5A5_A5A5] {
            let (a, b) = pair(words, seed);
            let want = hamming_scalar(&a, &b);
            let got = unsafe { hamming_x86_64::hamming_shuffle(&a, &b) };
            assert_eq!(got, want, "shuffle mismatch words={words} seed={seed:#x}");
        }
    }
}

#[test]
fn popcnt_handles_extreme_inputs() {
    if !avx2_available() {
        return;
    }
    // All-zero, all-ones, and complementary patterns exercise the
    // extremes of the popcount accumulator without RNG noise.
    let a = vec![0u64; 12];
    let b = vec![0u64; 12];
    assert_eq!(unsafe { hamming_x86_64::hamming_popcnt(&a, &b) }, 0);
    let a = vec![u64::MAX; 12];
    let b = vec![0u64; 12];
    assert_eq!(unsafe { hamming_x86_64::hamming_popcnt(&a, &b) }, 12 * 64);
    let a = vec![0x5555_5555_5555_5555u64; 12];
    let b = vec![0xAAAA_AAAA_AAAA_AAAAu64; 12];
    assert_eq!(unsafe { hamming_x86_64::hamming_popcnt(&a, &b) }, 12 * 64);
}

#[test]
fn shuffle_handles_extreme_inputs() {
    if !avx2_available() {
        return;
    }
    let a = vec![0u64; 12];
    let b = vec![0u64; 12];
    assert_eq!(unsafe { hamming_x86_64::hamming_shuffle(&a, &b) }, 0);
    let a = vec![u64::MAX; 12];
    let b = vec![0u64; 12];
    assert_eq!(unsafe { hamming_x86_64::hamming_shuffle(&a, &b) }, 12 * 64);
    let a = vec![0x5555_5555_5555_5555u64; 12];
    let b = vec![0xAAAA_AAAA_AAAA_AAAAu64; 12];
    assert_eq!(unsafe { hamming_x86_64::hamming_shuffle(&a, &b) }, 12 * 64);
}

#[test]
fn shuffle_tail_exercises_remainder_loop() {
    // words=5 → 40 bytes → 1 chunk of 32 + 8-byte tail. Forces the
    // tail loop path that the canonical widths never hit.
    if !avx2_available() {
        return;
    }
    let (a, b) = pair(5, 0xDEAD_BEEF);
    let want = hamming_scalar(&a, &b);
    let got = unsafe { hamming_x86_64::hamming_shuffle(&a, &b) };
    assert_eq!(got, want);
}

#[test]
fn both_variants_agree_on_large_n() {
    if !avx2_available() {
        return;
    }
    // 10_000 independent samples per variant — catches reduction-
    // overflow bugs (each i8 lane in the shuffle path can saturate
    // at 256 after 32 chunks if the per-chunk SAD is hoisted).
    let mut prng = SplitMix64::new(0x0100_AB0B);
    for _ in 0..10_000 {
        let words = ((prng.next_u64() as usize) & 0x7F) | 1; // 1..128
        let (a, b) = {
            let a: Vec<u64> = (0..words).map(|_| prng.next_u64()).collect();
            let b: Vec<u64> = (0..words).map(|_| prng.next_u64()).collect();
            (a, b)
        };
        let s = hamming_scalar(&a, &b);
        let p = unsafe { hamming_x86_64::hamming_popcnt(&a, &b) };
        let q = unsafe { hamming_x86_64::hamming_shuffle(&a, &b) };
        assert_eq!(s, p, "popcnt drift at words={words}");
        assert_eq!(s, q, "shuffle drift at words={words}");
    }
}
