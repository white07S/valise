//! Criterion microbench for `asymmetric_dot_pairs` — the gather-bound
//! search hot path (§4.1 of `docs/X86_64_SIMD_PLAN.md`).
//!
//! Reports SIMD-dispatcher and scalar-reference timings per sweep size.
//! On aarch64 the dispatcher routes to NEON; on x86_64 today it falls
//! to scalar (Phase 3 will activate the AVX2 path through the same
//! dispatcher).
//!
//! Run: `cargo bench --features bench --bench asymmetric_dot`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use valise::simd_bench;

/// Tiny deterministic PRNG so bench inputs are identical across runs
/// and arches. SplitMix64 from `src/codec/prng.rs`, inlined here so
/// the bench binary stays decoupled from crate internals.
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

fn make_inputs(
    p: usize,
) -> (
    Vec<f32>,
    Vec<u32>,
    Vec<u32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
    Vec<f32>,
) {
    let dim = 2 * p;
    let n_amp = 32;
    let n_phase = 64;
    let mut prng = SplitMix64::new(0x1111_2222_3333_4444);
    let q_rot: Vec<f32> = (0..dim)
        .map(|_| {
            let u = (prng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
            (u * 2.0 - 1.0) as f32
        })
        .collect();
    let amp_idx: Vec<u32> = (0..p).map(|_| (prng.next_u64() as u32) % n_amp).collect();
    let phase_idx: Vec<u32> = (0..p).map(|_| (prng.next_u64() as u32) % n_phase).collect();
    let sigma: Vec<f32> = (0..p)
        .map(|i| (i as f32 + 1.0) / (p as f32) * 0.03 + 0.005)
        .collect();
    let levels: Vec<f32> = (0..n_amp as usize)
        .map(|i| (i as f32 + 0.5) / n_amp as f32 * 4.0)
        .collect();
    let cos_lut: Vec<f32> = (0..n_phase as usize)
        .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).cos())
        .collect();
    let sin_lut: Vec<f32> = (0..n_phase as usize)
        .map(|i| (i as f32 * std::f32::consts::TAU / n_phase as f32).sin())
        .collect();
    (q_rot, amp_idx, phase_idx, sigma, levels, cos_lut, sin_lut)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("asymmetric_dot_pairs");
    for &p in &[64_usize, 384, 1536] {
        let (q_rot, amp_idx, phase_idx, sigma, levels, cos_lut, sin_lut) = make_inputs(p);
        g.throughput(Throughput::Elements(p as u64));

        g.bench_with_input(BenchmarkId::new("simd", p), &p, |b, _| {
            b.iter(|| {
                simd_bench::asymmetric_dot_pairs(
                    std::hint::black_box(&q_rot),
                    std::hint::black_box(&amp_idx),
                    std::hint::black_box(&phase_idx),
                    std::hint::black_box(&sigma),
                    std::hint::black_box(&levels),
                    std::hint::black_box(&cos_lut),
                    std::hint::black_box(&sin_lut),
                )
            })
        });

        g.bench_with_input(BenchmarkId::new("scalar", p), &p, |b, _| {
            b.iter(|| {
                simd_bench::scalar_asymmetric_dot_pairs(
                    std::hint::black_box(&q_rot),
                    std::hint::black_box(&amp_idx),
                    std::hint::black_box(&phase_idx),
                    std::hint::black_box(&sigma),
                    std::hint::black_box(&levels),
                    std::hint::black_box(&cos_lut),
                    std::hint::black_box(&sin_lut),
                )
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
