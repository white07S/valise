//! Criterion microbench for `pair_magnitudes` — encode-side
//! `√(re² + im²)` over interleaved (re, im) pairs (§4.5 of
//! `docs/X86_64_SIMD_PLAN.md`).
//!
//! Encode-only kernel; not on the query hot path. Bench timings here
//! serve as a regression canary for encode throughput. Phase 6 will
//! activate the AVX2 path.
//!
//! Run: `cargo bench --features bench --bench pair_magnitudes`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use valise::simd_bench;

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

fn make_pairs(p: usize) -> Vec<f32> {
    let mut prng = SplitMix64::new(0xFADE_C0FFEE);
    (0..2 * p)
        .map(|_| {
            let u = (prng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            (u * 2.0 - 1.0) as f32
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("pair_magnitudes");
    for &p in &[64_usize, 384, 1536] {
        let pairs = make_pairs(p);
        let mut mags = vec![0.0_f32; p];
        g.throughput(Throughput::Elements(p as u64));

        g.bench_with_input(BenchmarkId::new("simd", p), &p, |b, _| {
            b.iter(|| {
                simd_bench::pair_magnitudes(
                    std::hint::black_box(&pairs),
                    std::hint::black_box(&mut mags),
                )
            })
        });

        g.bench_with_input(BenchmarkId::new("scalar", p), &p, |b, _| {
            b.iter(|| {
                simd_bench::scalar_pair_magnitudes(
                    std::hint::black_box(&pairs),
                    std::hint::black_box(&mut mags),
                )
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
