// Prints results for the reader; stdout is this target's output channel.
#![allow(clippy::print_stdout, clippy::print_stderr)]

//! Direct-Rust baseline mirroring `bindings/valise-py/bench/bench.py`, so the
//! Python-binding overhead can be isolated (run BOTH in --release for a fair
//! opt-level comparison). Same workload: n=10k, dim=64, deferred calibration
//! sample=2000, cosine text+vector collection, 1000 vector / hybrid searches,
//! 1000 gets.
//!
//!     cargo run --release --example binding_overhead
//!     cargo run --release --example binding_overhead -- 10000 64

use std::time::Instant;

use valise::db::{Calibrate, Fusion, Key, Record, Rerank, Schema, Search, Store, Vector};

/// Deterministic unit-norm random matrix (xorshift; values don't affect timing,
/// only the workload shape does).
fn matrix(n: usize, dim: usize) -> Vec<f32> {
    let mut state: u64 = 0x9E3779B97F4A7C15;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // map to [-1, 1)
        ((state >> 40) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
    };
    let mut m = vec![0.0f32; n * dim];
    for row in m.chunks_mut(dim) {
        for x in row.iter_mut() {
            *x = next();
        }
        let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    m
}

fn row(m: &[f32], i: usize, dim: usize) -> &[f32] {
    &m[i * dim..(i + 1) * dim]
}

fn print_row(op: &str, n: usize, total_s: f64) {
    let per_op_us = if n > 0 { total_s / n as f64 * 1e6 } else { 0.0 };
    let per_s = if total_s > 0.0 {
        n as f64 / total_s
    } else {
        0.0
    };
    println!(
        "{:<26}  {:<10}  {:<12.2}  {:<12.2}  {:<14}",
        op,
        n,
        total_s * 1e3,
        per_op_us,
        format!("{per_s:.0}"),
    );
}

fn main() -> valise::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10_000);
    let dim: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(64);
    let sample = n.min(2000);

    let dir = std::env::temp_dir().join(format!("valise_rsbench_{n}_{dim}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir)?;

    println!("valise RUST baseline  n={n}  dim={dim}");
    println!();
    println!(
        "{:<26}  {:<10}  {:<12}  {:<12}  {:<14}",
        "op", "n", "total_ms", "per_op_us", "per_s"
    );
    println!("{}", "-".repeat(82));

    let mat = matrix(n, dim);
    let keys: Vec<Key> = (0..n).map(|i| Key::from(format!("k{i}"))).collect();
    let text = "quantum retrieval document";

    let schema = || {
        Schema::new().text("body").vector(
            "dense",
            Vector::dim(dim as u32).calibrate(Calibrate::auto_sample(sample)),
        )
    };

    // ---- 1. Batch put loop + commit (stage vs commit barrier) ----
    let s = Store::create(dir.join("batch.vls"))?;
    s.collection("kb", schema())?;
    let mut w = s.writer_owned();
    // Mirror the binding's put_many inner loop EXACTLY: owned keys moved in
    // (no per-iter clone), Record built vector-then-text. Otherwise the Rust
    // baseline does strictly more per-row work (a String clone ×N) than the
    // binding and unfairly looks slower on this ~2.5 µs/op path.
    let batch_keys: Vec<Key> = keys.clone();
    let t0 = Instant::now();
    for (i, key) in batch_keys.into_iter().enumerate() {
        w.put(
            "kb",
            key,
            Record::new()
                .vector("dense", row(&mat, i, dim))
                .text("body", text),
        )?;
    }
    let stage_s = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    w.commit()?;
    let commit_s = t0.elapsed().as_secs_f64();
    drop(w);
    print_row("put_many stage", n, stage_s);
    print_row("put_many commit barrier", n, commit_s);
    print_row("put_many total", n, stage_s + commit_s);

    // ---- 2. Per-call put baseline (fresh store, same loop) ----
    let s2 = Store::create(dir.join("percall.vls"))?;
    s2.collection("kb", schema())?;
    let mut w2 = s2.writer_owned();
    let t0 = Instant::now();
    for i in 0..n {
        w2.put(
            "kb",
            keys[i].clone(),
            Record::new()
                .text("body", text)
                .vector("dense", row(&mat, i, dim)),
        )?;
    }
    let pc_stage = t0.elapsed().as_secs_f64();
    let t0 = Instant::now();
    w2.commit()?;
    let pc_commit = t0.elapsed().as_secs_f64();
    drop(w2);
    print_row("put per-call total", n, pc_stage + pc_commit);

    // ---- 3. Vector search k=10 ----
    let reader = s.reader();
    let qn = 1000;
    let t0 = Instant::now();
    for i in 0..qn {
        let q = row(&mat, i % n, dim);
        let _ = reader.search(
            "kb",
            Search::new()
                .vector_with("dense", q, Rerank::Fast)
                .top_k(10),
        )?;
    }
    print_row("vector search k=10", qn, t0.elapsed().as_secs_f64());

    // ---- 3b. Hybrid RRF search ----
    let t0 = Instant::now();
    for i in 0..qn {
        let q = row(&mat, i % n, dim);
        let _ = reader.search(
            "kb",
            Search::new()
                .text("body", "quantum")
                .vector_with("dense", q, Rerank::Fast)
                .fuse(Fusion::rrf(60))
                .top_k(10),
        )?;
    }
    print_row("hybrid RRF search k=10", qn, t0.elapsed().as_secs_f64());

    // ---- 4. get ----
    let gn = 1000;
    let t0 = Instant::now();
    for i in 0..gn {
        let _ = reader.get("kb", &keys[i % n])?;
    }
    print_row("get", gn, t0.elapsed().as_secs_f64());

    let _ = std::fs::remove_dir_all(&dir);
    println!();
    println!("done.");
    Ok(())
}
