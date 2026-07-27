//! Criterion microbench for the fused block-Hadamard forward/inverse
//! kernels (§4.4 of `docs/X86_64_SIMD_PLAN.md`).
//!
//! Times the per-arch dispatcher (aarch64 → NEON; otherwise → scalar)
//! against the scalar reference. Phase 5 will activate AVX2 in the
//! dispatcher (with the `block_size >= 8` guard).
//!
//! Run: `cargo bench --features bench --bench fwht`

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

fn make_data(block_size: usize) -> (Vec<f32>, Vec<f32>) {
    // Production layout for d=768: dim = 3 × block_size when block=256;
    // we mirror that 3-block layout at each tested block_size to stress
    // both the per-block work and the across-block iteration cost.
    let dim = 3 * block_size;
    let mut prng = SplitMix64::new(0x0A_1B_2C_3D);
    let data: Vec<f32> = (0..dim)
        .map(|_| {
            let u = (prng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            (u * 2.0 - 1.0) as f32
        })
        .collect();
    let signs: Vec<f32> = (0..dim)
        .map(|_| if prng.next_u64() & 1 == 0 { 1.0 } else { -1.0 })
        .collect();
    (data, signs)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("block_hadamard");
    for &block_size in &[64_usize, 256, 1024] {
        let (data, signs) = make_data(block_size);
        g.throughput(Throughput::Elements(data.len() as u64));

        g.bench_with_input(
            BenchmarkId::new("forward_simd", block_size),
            &block_size,
            |b, &bs| {
                b.iter_batched_ref(
                    || data.clone(),
                    |buf| simd_bench::block_hadamard_forward(buf, &signs, bs),
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        g.bench_with_input(
            BenchmarkId::new("forward_scalar", block_size),
            &block_size,
            |b, &bs| {
                b.iter_batched_ref(
                    || data.clone(),
                    |buf| simd_bench::scalar_block_hadamard_forward(buf, &signs, bs),
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        g.bench_with_input(
            BenchmarkId::new("inverse_simd", block_size),
            &block_size,
            |b, &bs| {
                b.iter_batched_ref(
                    || data.clone(),
                    |buf| simd_bench::block_hadamard_inverse(buf, &signs, bs),
                    criterion::BatchSize::SmallInput,
                )
            },
        );

        g.bench_with_input(
            BenchmarkId::new("inverse_scalar", block_size),
            &block_size,
            |b, &bs| {
                b.iter_batched_ref(
                    || data.clone(),
                    |buf| simd_bench::scalar_block_hadamard_inverse(buf, &signs, bs),
                    criterion::BatchSize::SmallInput,
                )
            },
        );
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
