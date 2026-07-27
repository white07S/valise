// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Criterion microbench for the sign-sketch popcount-Hamming kernel
//! (§4.2 of `docs/X86_64_SIMD_PLAN.md`).
//!
//! On aarch64 `simd` is the NEON CNT path through the per-vector
//! dispatcher. On x86_64 we time the two AVX2 variants directly via
//! `hamming_avx2_popcnt` / `hamming_avx2_shuffle`; the dispatcher's
//! per-call `hamming` always falls to scalar on x86_64 because per
//! §4.2 the AVX2 path is reached via `scan_candidates_avx2` (whole-
//! scan target_feature wrap), not through the per-vector function.
//!
//! Run: `cargo bench --features bench --bench hamming`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use valise::sketch_bench;

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

fn make_pair(words: usize) -> (Vec<u64>, Vec<u64>) {
    let mut prng = SplitMix64::new(0xCAFEBABE_DEADBEEF);
    let a: Vec<u64> = (0..words).map(|_| prng.next_u64()).collect();
    let b: Vec<u64> = (0..words).map(|_| prng.next_u64()).collect();
    (a, b)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("hamming");
    for &words in &[8_usize, 12, 16, 32, 64] {
        let (a, b) = make_pair(words);
        g.throughput(Throughput::Bytes((words * 8) as u64));

        g.bench_with_input(BenchmarkId::new("simd", words), &words, |bn, _| {
            bn.iter(|| sketch_bench::hamming(std::hint::black_box(&a), std::hint::black_box(&b)))
        });

        g.bench_with_input(BenchmarkId::new("scalar", words), &words, |bn, _| {
            bn.iter(|| {
                sketch_bench::hamming_scalar(std::hint::black_box(&a), std::hint::black_box(&b))
            })
        });

        // Phase 2: direct AVX2 variants — only reached via
        // `scan_candidates_avx2` in production but timed in isolation
        // here so we can compare against the scalar baseline and pick
        // the threshold const.
        #[cfg(target_arch = "x86_64")]
        if std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("popcnt") {
            g.bench_with_input(BenchmarkId::new("avx2_popcnt", words), &words, |bn, _| {
                bn.iter(|| unsafe {
                    sketch_bench::hamming_avx2_popcnt(
                        std::hint::black_box(&a),
                        std::hint::black_box(&b),
                    )
                })
            });

            g.bench_with_input(BenchmarkId::new("avx2_shuffle", words), &words, |bn, _| {
                bn.iter(|| unsafe {
                    sketch_bench::hamming_avx2_shuffle(
                        std::hint::black_box(&a),
                        std::hint::black_box(&b),
                    )
                })
            });
        }
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
