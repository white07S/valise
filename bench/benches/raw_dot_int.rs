//! Criterion microbench for `QamSlidingEngine::raw_dot_int` — the
//! i8-SDOT rerank kernel (§4.3 of `docs/X86_64_SIMD_PLAN.md`).
//!
//! Times the dispatcher (aarch64 → NEON SDOT; otherwise → scalar
//! reference that mirrors NEON's `>>7` rounding-narrow) against the
//! scalar reference directly. Phase 4 will land the AVX2 variant in
//! the dispatcher.
//!
//! Run: `cargo bench --features bench --bench raw_dot_int`

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use valise::{QamLloydMaxBench, SlidingBench, SlidingPrep};

fn build_bench(num_pairs: usize) -> (SlidingBench, SlidingPrep, Vec<u8>) {
    let dim = 2 * num_pairs;
    // block_size must divide dim and be a power of two. Production
    // uses 1024 at dim=3072 and 256 at dim=768; for smaller dims we
    // pick the largest power-of-two that divides the dim, capped at
    // 1024.
    let mut block_size = 1024;
    while !dim.is_multiple_of(block_size) && block_size > 1 {
        block_size /= 2;
    }
    let mut codec = QamLloydMaxBench::with_config(dim, block_size, 5, 6, true).expect("codec");
    // Sample for calibration: 256 random unit vectors built from a
    // tiny PRNG. The calibrate path needs roughly this many to find a
    // sane σ profile.
    let mut prng = 0xA1B2_C3D4_5566_7788_u64;
    let mut nx = || {
        prng = prng.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((prng >> 33) as i32 as f32) * (1.0 / (i32::MAX as f32))
    };
    let sample: Vec<Vec<f32>> = (0..256)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| nx()).collect();
            let n2: f32 = v.iter().map(|&x| x * x).sum();
            let inv = 1.0 / n2.sqrt().max(1e-12);
            v.iter_mut().for_each(|x| *x *= inv);
            v
        })
        .collect();
    codec.calibrate(&sample).expect("calibrate");
    let sliding = codec.sliding().expect("sliding");

    // Encode one base vector and prepare one query.
    let base_vec = sample[0].clone();
    let base = codec.encode(&base_vec).expect("encode");
    let prep = sliding.prepare(&sample[1]).expect("prepare");
    (sliding, prep, base)
}

fn bench(c: &mut Criterion) {
    let mut g = c.benchmark_group("raw_dot_int");
    for &n in &[64_usize, 384, 1536] {
        // num_pairs must be multiple of 8 (QAM (5,6) kernel constraint).
        let pairs = (n / 8) * 8;
        if pairs == 0 {
            continue;
        }
        let (sliding, prep, base) = build_bench(pairs);
        g.throughput(Throughput::Elements(pairs as u64));

        g.bench_with_input(BenchmarkId::new("simd", pairs), &pairs, |b, _| {
            b.iter(|| sliding.raw_dot_int(std::hint::black_box(&prep), std::hint::black_box(&base)))
        });

        g.bench_with_input(BenchmarkId::new("scalar", pairs), &pairs, |b, _| {
            b.iter(|| {
                sliding.raw_dot_int_scalar(std::hint::black_box(&prep), std::hint::black_box(&base))
            })
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
