//! Scale-ceiling Valise vector benchmark.
//!
//! Builds one vector-only `.vls` store per corpus size `N` in `--scales`
//! and measures where each pipeline stage stops scaling:
//!
//! 1. **corpus** — `N ≤ corpus_len` uses the real dataset prefix;
//!    larger `N` synthesizes a TIMING corpus by tiling the real corpus
//!    with per-copy Gaussian noise (`σ = 0.01 × per-dim std`, seeded
//!    SplitMix64 + Box-Muller — no time randomness), normalized exactly
//!    like the loader normalizes real rows. Synthetic points are marked
//!    `"synthetic": true` and get **no recall numbers** (the noise-tiled
//!    corpus has no meaningful ground truth).
//! 2. **build** — `register_codec_*_from_sample` (first 4 096 real
//!    rows) → ingest → commit, mirroring `valise-e2e-bench`'s vector phase.
//! 3. **open decomposition** — a fresh child process times
//!    `ValiseFile::open` (header + catalog + mmap) and reads the engine's
//!    structured `VectorOpenProfile` (`VALISE_QUERY_PROFILE`) for the lazy
//!    vector-base build: per-family cache build (QAM inverse-norms /
//!    UPQ decoded-i8 rows) + sign-sketch derivation.
//! 4. **query passes** — warmup + `--nq` queries at
//!    `VectorFidelity::Full` in TWO re-exec'd child configurations:
//!    *parallel* (default rayon pool) and *single-thread*
//!    (`RAYON_NUM_THREADS=1`; rayon pools are process-global, so the
//!    single-thread config must be a fresh process). Each child reports
//!    per-stage p50/p95 aggregated from the engine's per-query
//!    [`valise::QueryProfile`]s (sketch scan → integer rerank → exact
//!    pass), total p50/p95, CPU seconds, effective cores, and its
//!    getrusage peak RSS (the parallel child doubles as the
//!    fresh-process RSS probe — same self-re-exec pattern as
//!    `valise-e2e-bench`).
//! 5. **derived** — per-stage bytes touched and effective GB/s:
//!    stage 1 = scanned × sketch-row bytes, stage 2 = reranked ×
//!    codec row bytes (QAM: packed codes; UPQ: decoded-i8 cache row),
//!    stage 3 = survivors × packed code bytes.
//!
//! Real-`N` points also get recall@10/100 against brute-force cosine
//! ground truth, cached under `bench/cache/` with the same naming as
//! `valise-e2e-bench` so existing caches hit.
//!
//! Stores are deleted between scales (10M × 768d ≈ 5.6 GB each); free
//! disk is checked before every scale and the scale is skipped with a
//! clear message when insufficient.
//!
//! ## Reproduce
//!
//! ```bash
//! cargo build --release -p valise-bench --bin valise-scale-bench
//! target/release/valise-scale-bench \
//!     --dataset-dir bench/datasets/cohere-medium-1m-f32 \
//!     --codec upq \
//!     --scales 10000,100000,1000000 \
//!     --nq 1000 --k 100 --channel-k 0 \
//!     --out bench/results/scale.json
//! ```

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use serde::{Deserialize, Serialize};

use valise::{
    AutoPromote, CreateOptions, Dtype, DtypeSet, EmbeddingSpaceSpec, OpenMode, PutFrame, PutVector,
    UpqDesignSource, ValiseFile, VectorContract, VectorFidelity, VectorMetric, VectorSearchQuery,
    last_query_profile, last_vector_open_profile,
};

/// Hidden env var switching this binary into the query-pass child mode
/// (same self-re-exec pattern as `valise-e2e-bench`'s RSS probe).
const CHILD_ENV: &str = "VALISE_SCALE_BENCH_CHILD";

/// Seed for the synthetic-corpus noise stream. Fixed — two runs at the
/// same `N` ingest bit-identical corpora.
const SYNTH_SEED: u64 = 0x005e_edca_110f_5ca1;

/// Rows synthesized per rayon batch during synthetic ingest.
const SYNTH_CHUNK_ROWS: usize = 8192;

#[derive(Parser)]
#[command(
    name = "valise-scale-bench",
    about = "Valise scale-ceiling bench: per-stage query profile / open decomposition / RSS across corpus sizes"
)]
struct Cli {
    /// Dataset directory (`corpus.f32`, `queries.f32`, `meta.json`).
    #[arg(long)]
    dataset_dir: PathBuf,
    /// Vector codec: `qam` = QAM Lloyd-Max (5,6); `upq` = unrestricted
    /// polar quantization (`--upq-cells`, Empirical ring design).
    #[arg(long, value_enum)]
    codec: CodecChoice,
    /// Comma-separated corpus sizes. Sizes beyond the dataset's
    /// `corpus_len` are synthesized (timing-only, no recall).
    #[arg(long, default_value = "10000,100000,1000000")]
    scales: String,
    /// Timed queries per configuration.
    #[arg(long, default_value_t = 1000)]
    nq: usize,
    /// Top-k per query.
    #[arg(long, default_value_t = 100)]
    k: usize,
    /// Sketch candidate budget. `0` = the engine's production default
    /// (`max(4·k, 2048)`), same convention as `valise-e2e-bench
    /// --vector-channel-k 0`; any other value is used verbatim.
    #[arg(long, default_value_t = 0)]
    channel_k: usize,
    /// Warm-up queries before each timed pass (cycles the query set).
    #[arg(long, default_value_t = 32)]
    warmup: usize,
    /// Directory holding `queries.f32` (defaults to `--dataset-dir`).
    #[arg(long)]
    queries_from: Option<PathBuf>,
    /// Working directory for the per-scale `.vls` stores.
    #[arg(long, default_value = "/tmp/valise-scale-bench")]
    work_dir: PathBuf,
    /// Run the query children WITHOUT `VALISE_QUERY_PROFILE` (per-stage
    /// fields become null). For the profiling-overhead A/B check.
    #[arg(long, default_value_t = false)]
    no_profile: bool,
    /// Output JSON path.
    #[arg(long)]
    out: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
enum CodecChoice {
    Qam,
    Upq,
}

impl CodecChoice {
    fn as_str(self) -> &'static str {
        match self {
            CodecChoice::Qam => "qam",
            CodecChoice::Upq => "upq",
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DatasetMetric {
    Cosine,
    L2,
}

impl DatasetMetric {
    fn as_str(self) -> &'static str {
        match self {
            DatasetMetric::Cosine => "cosine",
            DatasetMetric::L2 => "l2",
        }
    }
}

// ---- Report shapes ---------------------------------------------------------

#[derive(Serialize)]
struct Report {
    config: ConfigReport,
    points: Vec<ScalePoint>,
}

#[derive(Serialize)]
struct ConfigReport {
    dataset: String,
    dataset_dir: String,
    queries_from: String,
    codec: String,
    dim: usize,
    metric: String,
    l2_normalized_surrogate: bool,
    scales: Vec<usize>,
    nq: usize,
    k: usize,
    /// `null` = engine production default (`--channel-k 0`).
    channel_k: Option<usize>,
    warmup: usize,
    query_profile: bool,
    corpus_len: usize,
}

#[derive(Serialize)]
struct ScalePoint {
    n: usize,
    synthetic: bool,
    /// Wall seconds spent generating synthetic rows (excluded from `ingest_s`).
    synth_s: f64,
    ingest_s: f64,
    commit_s: f64,
    file_bytes: u64,
    open: OpenReport,
    parallel: ConfigStats,
    single_thread: ConfigStats,
    recall_at_10: Option<f64>,
    recall_at_100: Option<f64>,
    /// Peak RSS of the fresh parallel query child (mmap + derived
    /// caches + query overhead — no corpus, no GT).
    peak_rss_fresh_bytes: u64,
}

/// Open decomposition, measured in the fresh parallel child:
/// `total_ms = mmap_catalog_ms (the ValiseFile::open call) + vector_base_ms`
/// (the lazy first-search build), with the build split into cache build
/// and sketch derivation by the engine's `VectorOpenProfile`.
#[derive(Serialize)]
struct OpenReport {
    total_ms: f64,
    mmap_catalog_ms: f64,
    vector_base_ms: Option<f64>,
    sketch_derive_ms: Option<f64>,
    cache_build_ms: Option<f64>,
}

#[derive(Serialize)]
struct ConfigStats {
    #[serde(flatten)]
    child: ChildReport,
    stage_gbps: Option<StageGbps>,
}

/// Effective per-stage bandwidth: analytic bytes touched / stage p50.
#[derive(Serialize)]
struct StageGbps {
    stage1: Option<f64>,
    stage2: Option<f64>,
    stage3: Option<f64>,
}

/// One query-pass child's JSON line (also embedded verbatim in the
/// parent report). Per-stage fields are `null` when the child ran
/// without `VALISE_QUERY_PROFILE`.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct ChildReport {
    open_call_ms: f64,
    vb_build_ms: Option<f64>,
    sketch_derive_ms: Option<f64>,
    cache_build_ms: Option<f64>,
    prep_p50_us: Option<f64>,
    prep_p95_us: Option<f64>,
    stage1_p50_us: Option<f64>,
    stage1_p95_us: Option<f64>,
    stage2_p50_us: Option<f64>,
    stage2_p95_us: Option<f64>,
    stage3_p50_us: Option<f64>,
    stage3_p95_us: Option<f64>,
    /// p50 of per-query `prep + stage1 + stage2 + stage3` — compare
    /// against `total_p50_us` to spot untimed gaps.
    stage_sum_p50_us: Option<f64>,
    total_p50_us: f64,
    total_p95_us: f64,
    profiled_queries: usize,
    candidates_scanned_mean: Option<f64>,
    candidates_reranked_mean: Option<f64>,
    survivors_mean: Option<f64>,
    cpu_s: f64,
    wall_s: f64,
    effective_cores: f64,
    peak_rss_bytes: u64,
}

// ---- Shared helpers (same shapes as valise-e2e-bench) -------------------------

/// Process CPU seconds (user+sys) and peak RSS in bytes, via `getrusage`.
/// macOS reports `ru_maxrss` in bytes; Linux in KiB.
fn rusage_now() -> (f64, u64) {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        let cpu = ru.ru_utime.tv_sec as f64
            + ru.ru_utime.tv_usec as f64 * 1e-6
            + ru.ru_stime.tv_sec as f64
            + ru.ru_stime.tv_usec as f64 * 1e-6;
        let maxrss = ru.ru_maxrss as u64;
        let maxrss = if cfg!(target_os = "linux") {
            maxrss * 1024
        } else {
            maxrss
        };
        (cpu, maxrss)
    }
}

fn percentile_us(samples: &[f64], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut us = samples.to_vec();
    us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let i = ((p / 100.0) * (us.len() as f64 - 1.0)).round() as usize;
    us[i.min(us.len() - 1)]
}

fn load_f32_prefix(path: &Path, n_floats: usize) -> Result<Vec<f32>> {
    let mut f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let need = n_floats.checked_mul(4).ok_or_else(|| anyhow!("overflow"))?;
    let mut buf = vec![0u8; need];
    f.read_exact(&mut buf)
        .with_context(|| format!("read {n_floats} f32 from {path:?}"))?;
    let mut out = vec![0f32; n_floats];
    for (i, slot) in out.iter_mut().enumerate() {
        let off = i * 4;
        *slot = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
    }
    Ok(out)
}

fn l2_normalize_rows(data: &mut [f32], dim: usize) {
    use rayon::prelude::*;
    data.par_chunks_mut(dim).for_each(|row| {
        let norm = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
        let inv = 1.0 / norm;
        for x in row {
            *x *= inv;
        }
    });
}

#[derive(Deserialize)]
struct VectorMeta {
    dim: usize,
    corpus_len: usize,
    query_len: usize,
    #[serde(default)]
    metric: Option<String>,
}

fn dataset_metric(name: &str, meta: &VectorMeta) -> Result<DatasetMetric> {
    if let Some(m) = meta.metric.as_deref() {
        return match m {
            "cosine" => Ok(DatasetMetric::Cosine),
            "l2" => Ok(DatasetMetric::L2),
            other => Err(anyhow!(
                "dataset {name}: meta.json metric {other:?} unsupported (cosine | l2)"
            )),
        };
    }
    if name.starts_with("cohere") || name.starts_with("openai") {
        return Ok(DatasetMetric::Cosine);
    }
    if name.starts_with("sift") || name.starts_with("gist") {
        return Ok(DatasetMetric::L2);
    }
    Err(anyhow!(
        "dataset {name}: metric unknown — add a \"metric\" field (\"cosine\" | \"l2\") to meta.json"
    ))
}

// ---- Deterministic Gaussian noise (SplitMix64 + Box-Muller) ----------------

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Uniform in the open interval (0, 1) — never 0, so `ln` is safe.
fn unit_open(state: &mut u64) -> f64 {
    (((splitmix64(state) >> 11) as f64) + 0.5) / 9_007_199_254_740_992.0
}

fn gauss_pair(state: &mut u64) -> (f32, f32) {
    let u1 = unit_open(state);
    let u2 = unit_open(state);
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = std::f64::consts::TAU * u2;
    ((r * theta.cos()) as f32, (r * theta.sin()) as f32)
}

// ---- Dataset ---------------------------------------------------------------

struct ScaleDataset {
    name: String,
    dir: PathBuf,
    dim: usize,
    metric: DatasetMetric,
    corpus_len: usize,
    /// Real rows loaded (`min(max_scale, corpus_len)`), already
    /// normalized when `metric == L2`.
    real_rows: usize,
    corpus: Vec<f32>,
    nq: usize,
    queries: Vec<f32>,
    queries_path: PathBuf,
    /// Per-dim noise sigma (`0.01 × per-dim std` of the real corpus).
    /// Empty unless a synthetic scale was requested.
    sigma: Vec<f32>,
}

fn load_scale_dataset(cli: &Cli, scales: &[usize]) -> Result<ScaleDataset> {
    let name = cli
        .dataset_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let meta_path = cli.dataset_dir.join("meta.json");
    let meta: VectorMeta = serde_json::from_reader(File::open(&meta_path)?)
        .with_context(|| format!("read {meta_path:?}"))?;
    let metric = dataset_metric(&name, &meta)?;
    let dim = meta.dim;

    let max_scale = *scales.iter().max().ok_or_else(|| anyhow!("no scales"))?;
    let real_rows = max_scale.min(meta.corpus_len);
    let mut corpus = load_f32_prefix(&cli.dataset_dir.join("corpus.f32"), real_rows * dim)?;

    let queries_dir = cli
        .queries_from
        .clone()
        .unwrap_or_else(|| cli.dataset_dir.clone());
    let queries_path = queries_dir.join("queries.f32");
    if queries_dir == cli.dataset_dir && cli.nq > meta.query_len {
        return Err(anyhow!(
            "--nq {} exceeds dataset query_len {}",
            cli.nq,
            meta.query_len
        ));
    }
    let mut queries = load_f32_prefix(&queries_path, cli.nq * dim)?;

    if metric == DatasetMetric::L2 {
        // Angular surrogate: L2 datasets are normalized at load and
        // searched under cosine — same handling as valise-e2e-bench.
        l2_normalize_rows(&mut corpus, dim);
        l2_normalize_rows(&mut queries, dim);
        println!("[load] {name}: native metric is L2 — corpus + queries L2-normalized at load");
    }
    println!(
        "[load] {name}: real_rows={real_rows} dim={dim} nq={} metric={} (corpus_len={})",
        cli.nq,
        metric.as_str(),
        meta.corpus_len
    );

    let needs_synth = scales.iter().any(|&n| n > meta.corpus_len);
    let sigma = if needs_synth {
        let t = Instant::now();
        let s = per_dim_sigma(&corpus, dim, real_rows);
        println!(
            "[synth] per-dim sigma (0.01 × std over {real_rows} real rows) in {:.2}s",
            t.elapsed().as_secs_f64()
        );
        s
    } else {
        Vec::new()
    };

    Ok(ScaleDataset {
        name,
        dir: cli.dataset_dir.clone(),
        dim,
        metric,
        corpus_len: meta.corpus_len,
        real_rows,
        corpus,
        nq: cli.nq,
        queries,
        queries_path,
        sigma,
    })
}

/// `0.01 × per-dim std` over the loaded real corpus.
fn per_dim_sigma(corpus: &[f32], dim: usize, rows: usize) -> Vec<f32> {
    use rayon::prelude::*;
    let (sum, sq) = corpus
        .par_chunks(dim * 4096)
        .fold(
            || (vec![0f64; dim], vec![0f64; dim]),
            |mut acc, block| {
                for row in block.chunks_exact(dim) {
                    for (d, &v) in row.iter().enumerate() {
                        let v = v as f64;
                        acc.0[d] += v;
                        acc.1[d] += v * v;
                    }
                }
                acc
            },
        )
        .reduce(
            || (vec![0f64; dim], vec![0f64; dim]),
            |mut a, b| {
                for d in 0..dim {
                    a.0[d] += b.0[d];
                    a.1[d] += b.1[d];
                }
                a
            },
        );
    let n = rows as f64;
    (0..dim)
        .map(|d| {
            let mean = sum[d] / n;
            let var = (sq[d] / n - mean * mean).max(0.0);
            (0.01 * var.sqrt()) as f32
        })
        .collect()
}

impl ScaleDataset {
    /// Fill `out` with global corpus row `i`: the real row for
    /// `i < real_rows`, otherwise real row `i % real_rows` + seeded
    /// Gaussian noise, normalized like the loader when the dataset's
    /// native metric required it.
    fn fill_row(&self, i: usize, out: &mut [f32]) {
        let base = (i % self.real_rows) * self.dim;
        let src = &self.corpus[base..base + self.dim];
        if i < self.real_rows {
            out.copy_from_slice(src);
            return;
        }
        let mut state = SYNTH_SEED ^ (i as u64).wrapping_mul(0xD6E8_FEB8_6659_FD93);
        let mut d = 0;
        while d < self.dim {
            let (g0, g1) = gauss_pair(&mut state);
            out[d] = src[d] + self.sigma[d] * g0;
            if d + 1 < self.dim {
                out[d + 1] = src[d + 1] + self.sigma[d + 1] * g1;
            }
            d += 2;
        }
        if self.metric == DatasetMetric::L2 {
            let norm = out.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
            let inv = 1.0 / norm;
            for x in out.iter_mut() {
                *x *= inv;
            }
        }
    }
}

// ---- Ground truth + recall (cache-compatible with valise-e2e-bench) -----------

struct GroundTruth {
    depth: usize,
    /// `nq × depth`, row-major.
    ids: Vec<u32>,
}

fn gt_cache_path(ds: &ScaleDataset, n: usize, depth: usize) -> Option<PathBuf> {
    let bench_dir = ds.dir.parent()?.parent()?;
    let label = match ds.metric {
        DatasetMetric::Cosine => "cosine",
        DatasetMetric::L2 => "l2norm-cosine",
    };
    Some(bench_dir.join("cache").join(format!(
        "gt_{}_n{}_nq{}_{}_k{}.u32",
        ds.name, n, ds.nq, label, depth
    )))
}

fn exact_ground_truth(ds: &ScaleDataset, n: usize) -> Result<GroundTruth> {
    let depth = 100.min(n);
    let cache = gt_cache_path(ds, n, depth);
    if let Some(path) = cache.as_ref()
        && let Ok(bytes) = fs::read(path)
        && bytes.len() == ds.nq * depth * 4
    {
        let ids: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        println!("[gt] loaded cache {}", path.display());
        return Ok(GroundTruth { depth, ids });
    }

    use rayon::prelude::*;
    let t = Instant::now();
    let dim = ds.dim;
    let inv_norms: Vec<f32> = ds.corpus[..n * dim]
        .par_chunks(dim)
        .map(|row| 1.0 / row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12))
        .collect();
    let per_query: Vec<Vec<u32>> = (0..ds.nq)
        .into_par_iter()
        .map(|qi| {
            let q = &ds.queries[qi * dim..(qi + 1) * dim];
            let mut scored: Vec<(f32, u32)> = (0..n)
                .map(|i| {
                    let row = &ds.corpus[i * dim..(i + 1) * dim];
                    let dot: f32 = q.iter().zip(row).map(|(a, b)| a * b).sum();
                    (dot * inv_norms[i], i as u32)
                })
                .collect();
            scored.select_nth_unstable_by(depth - 1, |a, b| b.0.total_cmp(&a.0));
            scored.truncate(depth);
            scored.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
            scored.into_iter().map(|(_, i)| i).collect()
        })
        .collect();
    let ids: Vec<u32> = per_query.into_iter().flatten().collect();
    println!(
        "[gt] brute-force exact top-{depth} over N={n} nq={} in {:.2}s",
        ds.nq,
        t.elapsed().as_secs_f64()
    );
    if let Some(path) = cache.as_ref() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut bytes = Vec::with_capacity(ids.len() * 4);
        for id in &ids {
            bytes.extend_from_slice(&id.to_le_bytes());
        }
        fs::write(path, bytes)?;
        println!("[gt] cached {}", path.display());
    }
    Ok(GroundTruth { depth, ids })
}

fn recall_at(pred: &[Vec<u32>], gt: &GroundTruth, k: usize) -> Option<f64> {
    if gt.depth < k || pred.is_empty() || pred.iter().any(|p| p.len() < k) {
        return None;
    }
    let mut sum = 0.0f64;
    for (qi, p) in pred.iter().enumerate() {
        let truth: HashSet<u32> = gt.ids[qi * gt.depth..qi * gt.depth + k]
            .iter()
            .copied()
            .collect();
        let hits = p[..k].iter().filter(|id| truth.contains(id)).count();
        sum += hits as f64 / k as f64;
    }
    Some(sum / pred.len() as f64)
}

// ---- Disk preflight --------------------------------------------------------

/// Free bytes on the filesystem holding `path` (statvfs), `None` on error.
fn available_disk_bytes(path: &Path) -> Option<u64> {
    use std::os::unix::ffi::OsStrExt;
    let c = std::ffi::CString::new(path.as_os_str().as_bytes()).ok()?;
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return None;
    }
    #[allow(clippy::unnecessary_cast)] // fsblkcnt_t width is platform-dependent
    Some(st.f_bavail as u64 * st.f_frsize as u64)
}

// ---- Main ------------------------------------------------------------------

fn main() -> Result<()> {
    if std::env::var_os(CHILD_ENV).is_some() {
        return child_main();
    }
    let cli = Cli::parse();
    fs::create_dir_all(&cli.work_dir)?;
    let scales: Vec<usize> = cli
        .scales
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&n: &usize| n > 0)
        .collect();
    if scales.is_empty() {
        return Err(anyhow!("--scales parsed to nothing: {:?}", cli.scales));
    }
    let ds = load_scale_dataset(&cli, &scales)?;
    let channel_k: Option<usize> = match cli.channel_k {
        0 => None,
        k => Some(k),
    };

    let mut points: Vec<ScalePoint> = Vec::with_capacity(scales.len());
    for &n in &scales {
        println!();
        println!(
            "==== N = {n}{} ====",
            if n > ds.corpus_len {
                " (synthetic)"
            } else {
                ""
            }
        );
        match run_scale(&cli, &ds, n, channel_k) {
            Ok(point) => {
                print_point(&point);
                points.push(point);
            }
            Err(e) => {
                eprintln!("[scale {n}] SKIPPED: {e:#}");
            }
        }
    }

    let report = Report {
        config: ConfigReport {
            dataset: ds.name.clone(),
            dataset_dir: cli.dataset_dir.display().to_string(),
            queries_from: ds.queries_path.display().to_string(),
            codec: cli.codec.as_str().to_string(),
            dim: ds.dim,
            metric: ds.metric.as_str().to_string(),
            l2_normalized_surrogate: ds.metric == DatasetMetric::L2,
            scales,
            nq: cli.nq,
            k: cli.k,
            channel_k,
            warmup: cli.warmup,
            query_profile: !cli.no_profile,
            corpus_len: ds.corpus_len,
        },
        points,
    };
    if let Some(out_path) = cli.out.as_ref() {
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out_path, serde_json::to_string_pretty(&report)?)?;
        println!();
        println!("[wrote] {}", out_path.display());
    }
    Ok(())
}

fn run_scale(
    cli: &Cli,
    ds: &ScaleDataset,
    n: usize,
    channel_k: Option<usize>,
) -> Result<ScalePoint> {
    let synthetic = n > ds.corpus_len;
    let dim = ds.dim;

    // Disk preflight: conservative bound of raw f32 corpus volume
    // (covers the store file + commit transients with wide margin).
    let required = (n as u64) * (dim as u64) * 4 + (2 << 30);
    if let Some(avail) = available_disk_bytes(&cli.work_dir)
        && avail < required
    {
        return Err(anyhow!(
            "insufficient disk at {}: {:.1} GiB available, ~{:.1} GiB required for N={n} — \
             free space or drop this scale",
            cli.work_dir.display(),
            avail as f64 / (1u64 << 30) as f64,
            required as f64 / (1u64 << 30) as f64
        ));
    }

    let store_path = cli.work_dir.join(format!("scale-{n}.vls"));
    if store_path.exists() {
        fs::remove_file(&store_path)?;
    }

    // ---- create + register (identical shape to valise-e2e-bench) ----
    let calib_rows = n.min(4096).min(ds.real_rows);
    let calib_sample: Vec<Vec<f32>> = (0..calib_rows)
        .map(|i| ds.corpus[i * dim..(i + 1) * dim].to_vec())
        .collect();
    let opts = CreateOptions {
        auto_promote: AutoPromote::default(),
        vector: VectorContract {
            max_dim: dim as u32,
            allowed_dtypes: DtypeSet::ALL,
        },
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(&store_path, opts)?;
    let collection = valise.create_collection("c")?;
    let codec_id = match cli.codec {
        CodecChoice::Qam => valise.register_codec_qam_from_sample(dim, &calib_sample)?,
        CodecChoice::Upq => valise.register_codec_upq_from_sample_with_options(
            dim,
            2048,
            UpqDesignSource::Empirical,
            &calib_sample,
        )?,
    };
    let space = valise.register_embedding_space(EmbeddingSpaceSpec {
        provider: "bench".into(),
        model: ds.name.clone(),
        dimension: dim as u32,
        metric: VectorMetric::Cosine,
        normalized: ds.metric == DatasetMetric::L2,
        dtype: Dtype::F32,
        primary_codec_id: Some(codec_id),
        secondary_codec_id: None,
    })?;

    // ---- ingest (chunked; synthetic rows generated with rayon) ----
    let mut row_by_vid: Option<HashMap<u64, u32>> = if synthetic {
        None
    } else {
        Some(HashMap::with_capacity(n))
    };
    let mut synth_s = 0.0f64;
    let t_ingest = Instant::now();
    let mut buf = vec![0f32; SYNTH_CHUNK_ROWS * dim];
    let mut i = 0usize;
    while i < n {
        let chunk = SYNTH_CHUNK_ROWS.min(n - i);
        if i + chunk <= ds.real_rows {
            // Pure-real chunk: ingest straight from the loaded corpus.
            for j in i..i + chunk {
                let frame = valise.put_frame(PutFrame::document(collection, b""))?;
                let vid = valise.put_vector(PutVector {
                    owner_frame_id: frame,
                    embedding_space_id: space,
                    values: &ds.corpus[j * dim..(j + 1) * dim],
                })?;
                if let Some(map) = row_by_vid.as_mut() {
                    map.insert(vid.0, j as u32);
                }
            }
        } else {
            use rayon::prelude::*;
            let t_synth = Instant::now();
            buf[..chunk * dim]
                .par_chunks_mut(dim)
                .enumerate()
                .for_each(|(j, row)| ds.fill_row(i + j, row));
            synth_s += t_synth.elapsed().as_secs_f64();
            for j in 0..chunk {
                let frame = valise.put_frame(PutFrame::document(collection, b""))?;
                let _ = valise.put_vector(PutVector {
                    owner_frame_id: frame,
                    embedding_space_id: space,
                    values: &buf[j * dim..(j + 1) * dim],
                })?;
            }
        }
        i += chunk;
    }
    let ingest_s = t_ingest.elapsed().as_secs_f64() - synth_s;
    println!("[ingest] {ingest_s:.2}s (+{synth_s:.2}s synthesis)");

    // ---- commit ----
    let t = Instant::now();
    valise.commit()?;
    let commit_s = t.elapsed().as_secs_f64();
    println!("[commit] {commit_s:.2}s");
    drop(valise);
    let file_bytes = fs::metadata(&store_path)?.len();

    // ---- recall (real-N points only; untimed ranked pass) ----
    let (recall_at_10, recall_at_100) = if let Some(row_by_vid) = row_by_vid.as_ref() {
        let gt = exact_ground_truth(ds, n)?;
        let valise = ValiseFile::open(&store_path, OpenMode::ReadOnly)?;
        let mut ranked: Vec<Vec<u32>> = Vec::with_capacity(ds.nq);
        for qi in 0..ds.nq {
            let hits = valise.vector_search(VectorSearchQuery {
                embedding_space_id: space,
                query: ds.queries[qi * dim..(qi + 1) * dim].to_vec(),
                k: cli.k,
                channel_k,
                collection_filter: None,
                fidelity: VectorFidelity::Full,
            })?;
            ranked.push(
                hits.iter()
                    .filter_map(|h| row_by_vid.get(&h.vector_id.0).copied())
                    .collect(),
            );
        }
        drop(valise);
        let r10 = recall_at(&ranked, &gt, 10);
        let r100 = recall_at(&ranked, &gt, 100);
        println!(
            "[recall] r@10={}  r@100={}",
            fmt_opt(r10, 3),
            fmt_opt(r100, 3)
        );
        (r10, r100)
    } else {
        println!("[recall] skipped (synthetic corpus)");
        (None, None)
    };

    // ---- query-pass children: parallel + single-thread ----
    let parallel = run_child(cli, ds, &store_path, channel_k, ChildConfig::Parallel)?;
    let single = run_child(cli, ds, &store_path, channel_k, ChildConfig::SingleThread)?;

    // ---- cleanup (stores get big) ----
    fs::remove_file(&store_path).with_context(|| format!("cleanup {}", store_path.display()))?;

    let open = OpenReport {
        total_ms: parallel.open_call_ms + parallel.vb_build_ms.unwrap_or(0.0),
        mmap_catalog_ms: parallel.open_call_ms,
        vector_base_ms: parallel.vb_build_ms,
        sketch_derive_ms: parallel.sketch_derive_ms,
        cache_build_ms: parallel.cache_build_ms,
    };
    let peak_rss_fresh_bytes = parallel.peak_rss_bytes;
    Ok(ScalePoint {
        n,
        synthetic,
        synth_s,
        ingest_s,
        commit_s,
        file_bytes,
        open,
        parallel: with_gbps(cli, dim, parallel),
        single_thread: with_gbps(cli, dim, single),
        recall_at_10,
        recall_at_100,
        peak_rss_fresh_bytes,
    })
}

/// Analytic bytes-touched per stage → effective GB/s at the stage p50.
fn with_gbps(cli: &Cli, dim: usize, child: ChildReport) -> ConfigStats {
    let sketch_row_bytes = dim.div_ceil(64) * 8;
    // Packed code bytes per row: 11 bits per (amp, phase) pair for both
    // QAM(5,6) and UPQ-2048 at the storage-equivalent operating point.
    let code_bytes = (dim / 2 * 11).div_ceil(8);
    // Stage 2 reads packed codes on the QAM path, decoded-i8 cache rows
    // (1 byte/dim) on the UPQ path.
    let stage2_row_bytes = match cli.codec {
        CodecChoice::Qam => code_bytes,
        CodecChoice::Upq => dim,
    };
    let gbps = |rows: Option<f64>, row_bytes: usize, p50_us: Option<f64>| -> Option<f64> {
        let rows = rows?;
        let p50 = p50_us?;
        (p50 > 0.0).then(|| rows * row_bytes as f64 / (p50 * 1e-6) / 1e9)
    };
    let stage_gbps = child.stage1_p50_us.is_some().then(|| StageGbps {
        stage1: gbps(
            child.candidates_scanned_mean,
            sketch_row_bytes,
            child.stage1_p50_us,
        ),
        stage2: gbps(
            child.candidates_reranked_mean,
            stage2_row_bytes,
            child.stage2_p50_us,
        ),
        stage3: gbps(child.survivors_mean, code_bytes, child.stage3_p50_us),
    });
    ConfigStats { child, stage_gbps }
}

// ---- Child orchestration ---------------------------------------------------

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ChildConfig {
    Parallel,
    SingleThread,
}

fn run_child(
    cli: &Cli,
    ds: &ScaleDataset,
    store: &Path,
    channel_k: Option<usize>,
    config: ChildConfig,
) -> Result<ChildReport> {
    let exe = std::env::current_exe().context("child: current_exe")?;
    let mut cmd = Command::new(exe);
    cmd.env(CHILD_ENV, "1")
        .arg(store)
        .arg(&ds.queries_path)
        .arg(ds.dim.to_string())
        .arg(ds.nq.to_string())
        .arg(cli.k.to_string())
        .arg(match channel_k {
            Some(k) => k.to_string(),
            None => "none".to_string(),
        })
        .arg(if ds.metric == DatasetMetric::L2 {
            "1"
        } else {
            "0"
        })
        .arg(cli.warmup.to_string());
    if cli.no_profile {
        cmd.env_remove("VALISE_QUERY_PROFILE");
    } else {
        cmd.env("VALISE_QUERY_PROFILE", "1");
    }
    match config {
        ChildConfig::Parallel => {
            cmd.env_remove("RAYON_NUM_THREADS");
        }
        ChildConfig::SingleThread => {
            cmd.env("RAYON_NUM_THREADS", "1");
        }
    }
    let label = match config {
        ChildConfig::Parallel => "parallel",
        ChildConfig::SingleThread => "single-thread",
    };
    let out = cmd
        .output()
        .with_context(|| format!("spawn {label} query child"))?;
    if !out.status.success() {
        return Err(anyhow!(
            "{label} query child exited with {}: {}",
            out.status,
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    let line = stdout
        .lines()
        .rev()
        .map(str::trim)
        .find(|l| l.starts_with('{'))
        .ok_or_else(|| anyhow!("{label} child: no JSON line on stdout: {stdout:?}"))?;
    let report: ChildReport =
        serde_json::from_str(line).with_context(|| format!("{label} child: parse JSON"))?;
    println!(
        "[{label}] total p50={:.0}µs p95={:.0}µs  stages p50 prep={} s1={} s2={} s3={}  \
         cores={:.2}  rss={:.1} MiB",
        report.total_p50_us,
        report.total_p95_us,
        fmt_opt(report.prep_p50_us, 0),
        fmt_opt(report.stage1_p50_us, 0),
        fmt_opt(report.stage2_p50_us, 0),
        fmt_opt(report.stage3_p50_us, 0),
        report.effective_cores,
        report.peak_rss_bytes as f64 / (1024.0 * 1024.0),
    );
    Ok(report)
}

// ---- Child side ------------------------------------------------------------

/// Positional args (parent-controlled, no clap):
/// `<store.vls> <queries.f32> <dim> <nq> <k> <channel_k|none> <normalize 0|1> <warmup>`.
///
/// Opens the store read-only (timed — this is the mmap + catalog half of
/// the open decomposition), warms up (the first search triggers the lazy
/// vector-base build whose `VectorOpenProfile` supplies the sketch/cache
/// split), then runs `nq` timed queries at `VectorFidelity::Full`,
/// draining `last_query_profile()` after each. Emits one JSON
/// [`ChildReport`] line on stdout.
fn child_main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 8 {
        return Err(anyhow!(
            "scale child: expected 8 args (store queries_f32 dim nq k channel_k normalize warmup), got {}",
            args.len()
        ));
    }
    let store = PathBuf::from(&args[0]);
    let queries_path = PathBuf::from(&args[1]);
    let dim: usize = args[2].parse().context("scale child: dim")?;
    let nq: usize = args[3].parse().context("scale child: nq")?;
    let k: usize = args[4].parse().context("scale child: k")?;
    let channel_k: Option<usize> = match args[5].as_str() {
        "none" => None,
        s => Some(s.parse().context("scale child: channel_k")?),
    };
    let normalize = args[6] == "1";
    let warmup: usize = args[7].parse().context("scale child: warmup")?;
    if nq == 0 {
        return Err(anyhow!("scale child: nq must be ≥ 1"));
    }
    let profiling = std::env::var_os("VALISE_QUERY_PROFILE").is_some();

    let mut queries = load_f32_prefix(&queries_path, nq * dim)?;
    if normalize {
        l2_normalize_rows(&mut queries, dim);
    }

    let t_open = Instant::now();
    let valise = ValiseFile::open(&store, OpenMode::ReadOnly)?;
    let open_call_ms = t_open.elapsed().as_secs_f64() * 1e3;
    let space = valise
        .embedding_spaces()
        .iter()
        .find(|s| s.dimension as usize == dim)
        .map(|s| s.embedding_space_id)
        .ok_or_else(|| anyhow!("scale child: no embedding space with dim {dim} in {store:?}"))?;
    let make_query = |q: &[f32]| VectorSearchQuery {
        embedding_space_id: space,
        query: q.to_vec(),
        k,
        channel_k,
        collection_filter: None,
        fidelity: VectorFidelity::Full,
    };

    // Warmup — the first call performs the lazy vector-base build
    // (base pointers + per-family caches + sketch derivation).
    for i in 0..warmup.max(1) {
        let qi = i % nq;
        let _ = valise.vector_search(make_query(&queries[qi * dim..(qi + 1) * dim]))?;
    }
    let vb = last_vector_open_profile();

    // Timed pass.
    let us = |d: Duration| d.as_secs_f64() * 1e6;
    let mut totals: Vec<f64> = Vec::with_capacity(nq);
    let mut preps: Vec<f64> = Vec::with_capacity(nq);
    let mut stage1: Vec<f64> = Vec::with_capacity(nq);
    let mut stage2: Vec<f64> = Vec::with_capacity(nq);
    let mut stage3: Vec<f64> = Vec::with_capacity(nq);
    let mut stage_sums: Vec<f64> = Vec::with_capacity(nq);
    let (mut scanned_sum, mut reranked_sum, mut survivors_sum) = (0f64, 0f64, 0f64);
    let (cpu0, _) = rusage_now();
    let wall0 = Instant::now();
    for qi in 0..nq {
        let t0 = Instant::now();
        let _ = valise.vector_search(make_query(&queries[qi * dim..(qi + 1) * dim]))?;
        totals.push(us(t0.elapsed()));
        if profiling && let Some(p) = last_query_profile() {
            preps.push(us(p.prep));
            stage1.push(us(p.stage1_sketch));
            stage2.push(us(p.stage2_rerank));
            stage3.push(us(p.stage3_exact));
            stage_sums.push(us(p.prep
                + p.stage1_sketch
                + p.stage2_rerank
                + p.stage3_exact));
            scanned_sum += p.candidates_scanned as f64;
            reranked_sum += p.candidates_reranked as f64;
            survivors_sum += p.survivors as f64;
        }
    }
    let wall_s = wall0.elapsed().as_secs_f64();
    let (cpu1, peak_rss_bytes) = rusage_now();
    let cpu_s = cpu1 - cpu0;

    let profiled = preps.len();
    let mean = |sum: f64| (profiled > 0).then(|| sum / profiled as f64);
    let pct = |v: &[f64], p: f64| (!v.is_empty()).then(|| percentile_us(v, p));
    let report = ChildReport {
        open_call_ms,
        vb_build_ms: vb.map(|p| p.total.as_secs_f64() * 1e3),
        sketch_derive_ms: vb.map(|p| p.sketch_derive.as_secs_f64() * 1e3),
        cache_build_ms: vb.map(|p| p.cache_build.as_secs_f64() * 1e3),
        prep_p50_us: pct(&preps, 50.0),
        prep_p95_us: pct(&preps, 95.0),
        stage1_p50_us: pct(&stage1, 50.0),
        stage1_p95_us: pct(&stage1, 95.0),
        stage2_p50_us: pct(&stage2, 50.0),
        stage2_p95_us: pct(&stage2, 95.0),
        stage3_p50_us: pct(&stage3, 50.0),
        stage3_p95_us: pct(&stage3, 95.0),
        stage_sum_p50_us: pct(&stage_sums, 50.0),
        total_p50_us: percentile_us(&totals, 50.0),
        total_p95_us: percentile_us(&totals, 95.0),
        profiled_queries: profiled,
        candidates_scanned_mean: mean(scanned_sum),
        candidates_reranked_mean: mean(reranked_sum),
        survivors_mean: mean(survivors_sum),
        cpu_s,
        wall_s,
        effective_cores: if wall_s > 0.0 { cpu_s / wall_s } else { 0.0 },
        peak_rss_bytes,
    };
    println!("{}", serde_json::to_string(&report)?);
    Ok(())
}

// ---- Console output --------------------------------------------------------

fn fmt_opt(v: Option<f64>, decimals: usize) -> String {
    match v {
        Some(v) => format!("{v:.decimals$}"),
        None => "—".to_string(),
    }
}

fn print_point(p: &ScalePoint) {
    println!(
        "[open] total={:.1}ms (mmap+catalog={:.1}ms, vector_base={}ms; sketch={}ms cache={}ms)",
        p.open.total_ms,
        p.open.mmap_catalog_ms,
        fmt_opt(p.open.vector_base_ms, 1),
        fmt_opt(p.open.sketch_derive_ms, 1),
        fmt_opt(p.open.cache_build_ms, 1),
    );
    println!(
        "[store] {:.1} MiB on disk  fresh reader RSS {:.1} MiB",
        p.file_bytes as f64 / (1024.0 * 1024.0),
        p.peak_rss_fresh_bytes as f64 / (1024.0 * 1024.0),
    );
    for (label, c) in [
        ("parallel", &p.parallel),
        ("single-thread", &p.single_thread),
    ] {
        let g = c.stage_gbps.as_ref();
        println!(
            "[{label}] stage-sum p50={}µs vs total p50={:.0}µs  GB/s s1={} s2={} s3={}",
            fmt_opt(c.child.stage_sum_p50_us, 0),
            c.child.total_p50_us,
            fmt_opt(g.and_then(|g| g.stage1), 1),
            fmt_opt(g.and_then(|g| g.stage2), 2),
            fmt_opt(g.and_then(|g| g.stage3), 2),
        );
    }
}
