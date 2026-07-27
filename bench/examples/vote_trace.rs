//! Concurrency before/after: single-query p50/p95 + recall, then a
//! reader-thread sweep (qps / speedup / per-thread p50/p95) at the
//! production ck = N/4. Mirrors REPRODUCE.md's concurrent-search phase.
//!
//! Run: cargo run --release --example vote_trace -- [N] [nq] [LABEL]

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use rayon::prelude::*;
use valise::{
    AutoPromote, CreateOptions, Database, Dtype, DtypeSet, EmbeddingSpaceSpec, OpenMode, PutFrame,
    PutVector, QamLloydMaxBench, QamLloydMaxParams, ValiseFile, VectorContract, VectorFidelity,
    VectorId, VectorMetric, VectorSearchQuery,
};

const DIM: usize = 768;
const DATASET: &str = "bench/datasets/cohere-medium-1m-f32";

fn load_f32_prefix(path: &PathBuf, n: usize) -> Result<Vec<f32>> {
    use std::io::Read;
    let mut f = std::fs::File::open(path).with_context(|| format!("open {path:?}"))?;
    let mut buf = vec![0u8; n * 4];
    f.read_exact(&mut buf)?;
    let mut out = vec![0f32; n];
    for (i, s) in out.iter_mut().enumerate() {
        *s = f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
    }
    Ok(out)
}

fn pct(samples: &mut [f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let i = ((p / 100.0) * (samples.len() as f64 - 1.0)).round() as usize;
    samples[i.min(samples.len() - 1)]
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

/// Exact cosine top-k row indices for one query (brute force).
fn exact_topk(query: &[f32], corpus: &[f32], n: usize, k: usize) -> Vec<u32> {
    let qn: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mut sims: Vec<(f32, u32)> = (0..n)
        .into_par_iter()
        .map(|i| {
            let v = &corpus[i * DIM..(i + 1) * DIM];
            let mut dot = 0.0f32;
            let mut vn = 0.0f32;
            for d in 0..DIM {
                dot += query[d] * v[d];
                vn += v[d] * v[d];
            }
            let sim = if vn > 0.0 && qn > 0.0 {
                dot / (qn * vn.sqrt())
            } else {
                0.0
            };
            (sim, i as u32)
        })
        .collect();
    let kk = k.min(sims.len());
    sims.select_nth_unstable_by(kk - 1, |a, b| b.0.total_cmp(&a.0));
    sims.truncate(kk);
    sims.into_iter().map(|(_, i)| i).collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100_000);
    let nq: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(1000);
    let label = args.get(3).cloned().unwrap_or_else(|| "run".into());
    let top_k = 100;
    let ck = (n / 4).max(100);
    let recall_nq = nq.min(200);

    let base = PathBuf::from(DATASET);
    let corpus = load_f32_prefix(&base.join("corpus.f32"), n * DIM)?;
    let queries = load_f32_prefix(&base.join("queries.f32"), nq * DIM)?;
    println!("[load] N={n} dim={DIM} nq={nq} ck={ck} label={label}");

    let calib_rows = n.min(4096);
    let calib: Vec<Vec<f32>> = (0..calib_rows)
        .map(|i| corpus[i * DIM..(i + 1) * DIM].to_vec())
        .collect();
    let mut cb = QamLloydMaxBench::with_config(DIM, 256, 5, 6, true)?;
    cb.calibrate(&calib)?;
    let params: QamLloydMaxParams = cb.to_params([0u8; 32]);

    let path = PathBuf::from("/tmp/valise-vote-trace.vls");
    let _ = std::fs::remove_file(&path);
    let opts = CreateOptions {
        auto_promote: AutoPromote {
            non_f8_threshold: 50_000,
            f8_threshold: 250_000,
        },
        vector: VectorContract {
            max_dim: DIM as u32,
            allowed_dtypes: DtypeSet::ALL,
        },
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(&path, opts)?;
    let collection = valise.create_collection("c")?;
    let codec_id = valise.register_codec_qam(params)?;
    let space = valise.register_embedding_space(EmbeddingSpaceSpec {
        provider: "cohere".into(),
        model: "x".into(),
        dimension: DIM as u32,
        metric: VectorMetric::Cosine,
        normalized: false,
        dtype: Dtype::F32,
        primary_codec_id: Some(codec_id),
        secondary_codec_id: None,
    })?;
    let mut row_to_vid: Vec<VectorId> = Vec::with_capacity(n);
    let t = Instant::now();
    for i in 0..n {
        let v = &corpus[i * DIM..(i + 1) * DIM];
        let frame = valise.put_frame(PutFrame::document(collection, b""))?;
        let vid = valise.put_vector(PutVector {
            owner_frame_id: frame,
            embedding_space_id: space,
            values: v,
        })?;
        row_to_vid.push(vid);
    }
    let ingest_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    valise.commit()?;
    let commit_s = t.elapsed().as_secs_f64();
    drop(valise);

    let size_bytes = std::fs::metadata(&path)?.len();
    let size_mib = size_bytes as f64 / (1024.0 * 1024.0);
    let b_per_vec = size_bytes as f64 / n as f64;

    // ---- brute-force ground truth (recall_nq queries) ----
    let gt: Vec<std::collections::HashSet<VectorId>> = (0..recall_nq)
        .map(|qi| {
            let q = &queries[qi * DIM..(qi + 1) * DIM];
            exact_topk(q, &corpus, n, top_k)
                .into_iter()
                .map(|r| row_to_vid[r as usize])
                .collect()
        })
        .collect();

    // ---- single-thread table row (ck = N/4) ----
    let mk_query = |q: &[f32]| VectorSearchQuery {
        embedding_space_id: space,
        query: q.to_vec(),
        k: top_k,
        channel_k: Some(ck),
        collection_filter: None,
        fidelity: VectorFidelity::Full,
    };
    {
        let valise = ValiseFile::open(&path, OpenMode::ReadOnly)?;
        for i in 0..64 {
            let q = &queries[(i % nq) * DIM..(i % nq + 1) * DIM];
            let _ = valise.vector_search(mk_query(q))?;
        }
        let mut tot = Vec::with_capacity(nq);
        for qi in 0..nq {
            let q = &queries[qi * DIM..(qi + 1) * DIM];
            let t0 = Instant::now();
            let _ = valise.vector_search(mk_query(q))?;
            tot.push(us(t0.elapsed()));
        }
        let mut recall_sum = 0.0;
        for qi in 0..recall_nq {
            let q = &queries[qi * DIM..(qi + 1) * DIM];
            let hits = valise.vector_search(mk_query(q))?;
            let inter = hits
                .iter()
                .filter(|h| gt[qi].contains(&h.vector_id))
                .count();
            recall_sum += inter as f64 / top_k as f64;
        }
        println!(
            "\n  {:<22}  {:>8}  {:>8}  {:>8}  {:>6}  {:>8}  {:>8}  {:>10}",
            "config", "ingest s", "commit s", "size MiB", "B/vec", "p50 us", "p95 us", "recall@100"
        );
        println!(
            "  {:<22}  {:>8.3}  {:>8.3}  {:>8.2}  {:>6.1}  {:>8.0}  {:>8.0}  {:>10.4}",
            format!("{label} 1-thread"),
            ingest_s,
            commit_s,
            size_mib,
            b_per_vec,
            pct(&mut tot, 50.0),
            pct(&mut tot, 95.0),
            recall_sum / recall_nq as f64,
        );
    }

    // ---- concurrency sweep (shared Database, N reader threads) ----
    let queries = Arc::new(queries);
    let db = Database::open(&path, OpenMode::ReadOnly)?;
    println!(
        "\nconcurrency sweep (ck={ck}, fidelity=Full) — {label}:\n  {:>7}  {:>9}  {:>9}  {:>9}  {:>9}",
        "threads", "qps", "speedup", "p50 us", "p95 us"
    );
    let mut base_qps = 0.0;
    for &threads in &[1usize, 2, 4, 8] {
        let start = Instant::now();
        let mut handles = vec![];
        for tid in 0..threads {
            let db = db.clone();
            let queries = Arc::clone(&queries);
            handles.push(std::thread::spawn(move || -> (usize, f64, f64) {
                let reader = db.reader();
                let mut s = Vec::with_capacity(nq);
                let off = tid % nq.max(1);
                for i in 0..nq {
                    let qi = (off + i) % nq;
                    let q = &queries[qi * DIM..(qi + 1) * DIM];
                    let t0 = Instant::now();
                    let _ = reader.vector_search(VectorSearchQuery {
                        embedding_space_id: space,
                        query: q.to_vec(),
                        k: top_k,
                        channel_k: Some(ck),
                        collection_filter: None,
                        fidelity: VectorFidelity::Full,
                    });
                    s.push(us(t0.elapsed()));
                }
                (nq, pct(&mut s, 50.0), pct(&mut s, 95.0))
            }));
        }
        let mut total = 0;
        let mut p50sum = 0.0;
        let mut p95sum = 0.0;
        for h in handles {
            let (nn, p50, p95) = h.join().unwrap();
            total += nn;
            p50sum += p50;
            p95sum += p95;
        }
        let wall = start.elapsed().as_secs_f64();
        let qps = total as f64 / wall;
        if threads == 1 {
            base_qps = qps;
        }
        println!(
            "  {:>7}  {:>9.0}  {:>8.2}x  {:>9.0}  {:>9.0}",
            threads,
            qps,
            qps / base_qps,
            p50sum / threads as f64,
            p95sum / threads as f64,
        );
    }
    drop(db);
    let _ = std::fs::remove_file(&path);
    Ok(())
}
