//! End-to-end Valise benchmark.
//!
//! Builds two `.vls` files from real corpora — one **text-only** (BEIR
//! scifact, BM25), one **vector-only** (any `bench/datasets/<name>`
//! f32 corpus: Cohere d=768, OpenAI d=1536, GIST d=960, SIFT d=128 —
//! first N rows, `--codec qam` (5,6) or `--codec upq`) — and measures
//! every lifecycle phase the format goes through:
//!
//! 1. **ingest**  — `put_frame` / `put_vector` / `index_frame_text` loop
//! 2. **commit**  — durable flush of text segments and vector codec bytes
//! 3. **calibrate** — V-curve sweep over impact-vote-rerank tiers,
//!    picking the cheapest tier whose mean top-k overlap vs Exact is ≥
//!    `--text-calibrate-target` (with a 2σ binomial-noise margin). Text
//!    only.
//! 4. **search** — warm-up + N trials, p50/p95 µs across trials, plus
//!    recall@10 / recall@100 against exact ground truth (brute-force
//!    f32 over the ingested prefix, cached under `bench/cache/`) for
//!    Valise and the vector peers, so the comparison is recall-matched
//! 5. **storage** — `.vls` file size on disk after commit
//!
//! ## Metric handling
//!
//! Valise's sketch pipeline is **cosine end-to-end**: stage 1 is an
//! angular sign-sketch Hamming scan, the QAM(5,6) sliding stage-2
//! kernel scores `-dot·inv_norm` regardless of `space.metric`, and the
//! UPQ rerank path is hard-coded cosine. L2 datasets (SIFT/GIST) are
//! therefore run the way the cross-dataset experiments did
//! (`valise_experiments/harness/examples/upq_768_bench.rs::read_normalized`,
//! `valise_experiments/pareto/PARETO_RESEARCH_2026-06.md` Part 9): every
//! vector is **L2-normalized at load** and searched under cosine (an
//! angular surrogate — on unit vectors L2 and cosine rank identically).
//! Primary recall is measured against exact cosine ground truth over
//! the normalized, ingested prefix; when the official texmex `gt.u32`
//! (L2 over the raw, full corpus) is applicable — i.e. `--vector-n`
//! equals the full corpus — recall against it is also reported as a
//! secondary `recall_official_*` number.
//!
//! Result lands as one JSON document so a CI pipeline can diff
//! against a baseline. The working `.vls` files are deleted at the
//! end of the run.
//!
//! ## Reproduce
//!
//! ```bash
//! cargo build --release -p valise-bench --bin valise-e2e-bench
//! target/release/valise-e2e-bench \
//!     --beir-dir bench/beir-data/scifact \
//!     --vector-dir bench/datasets/cohere-medium-1m-f32 \
//!     --vector-n 100000 \
//!     --codec qam \
//!     --out bench/results/e2e.json
//! ```
//!
//! See `bench/REPRODUCE.md` for the full prerequisites + expected
//! numbers on an Apple M-series machine.

use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{BufRead, BufReader, Read},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use serde::Deserialize;

use std::sync::Arc;
use std::thread;
use valise::{
    AnalyzerDesc, AnalyzerId, AutoPromote, CreateOptions, Database, Dtype, DtypeSet,
    EmbeddingSpaceSpec, FieldDesc, FieldSchemaDesc, FieldSchemaId, FieldSource, IdfVariant,
    OpenMode, PunctuationPolicy, PutFrame, PutVector, QueryAlgorithm, RetrievalProfileDesc,
    RetrievalProfileId, RetrievalProfileParams, RetrievalProfileType, Stemming, StopwordsPolicy,
    TextQuery, TextSpaceDesc, TextSpaceId, Tokenizer, UnicodeNormalization, UpqDesignSource,
    ValiseFile, VectorContract, VectorFidelity, VectorMetric, VectorSearchQuery,
};

#[derive(Parser)]
#[command(
    name = "valise-e2e-bench",
    about = "End-to-end Valise bench: ingest → commit → calibrate → search → storage"
)]
struct Cli {
    /// BEIR-format dataset directory for the text file. Must contain
    /// `corpus.jsonl`, `queries.jsonl`, `qrels/test.tsv`.
    #[arg(long)]
    beir_dir: PathBuf,
    /// Cohere-style dataset directory for the vector file. Must
    /// contain `corpus.f32`, `queries.f32`, `meta.json`.
    #[arg(long)]
    vector_dir: PathBuf,
    /// Number of corpus rows to ingest from `vector_dir` (the first
    /// `--vector-n` × dim f32s are used).
    #[arg(long, default_value_t = 100_000)]
    vector_n: usize,
    /// Number of query rows to time. Must be ≤ `meta.json::query_len`.
    #[arg(long, default_value_t = 1000)]
    vector_nq: usize,
    /// Vector codec for the embedding space: `qam` = production QAM
    /// Lloyd-Max (5,6); `upq` = unrestricted polar quantization
    /// (`--upq-cells`, Empirical ring design). Both are calibrated on
    /// the first 4 096 corpus rows and searched through the normal
    /// `vector_search` path (ingest → commit → open → sketch → rerank).
    #[arg(long, value_enum, default_value_t = CodecChoice::Qam)]
    codec: CodecChoice,
    /// UPQ cell budget (codebook size). 2048 cells = 11 bits/pair =
    /// 5.5 bits/dim — the storage-equivalent of QAM (5,6).
    #[arg(long, default_value_t = 2048)]
    upq_cells: u32,
    /// Vector candidate budget (`channel_k`) for the sketch stage.
    /// Unset: the legacy `N/4` operating point (recall-ceiling mode).
    /// `0`: the engine's production default budget (`max(4·k, 2048)`).
    /// Any other value is used verbatim.
    #[arg(long)]
    vector_channel_k: Option<usize>,
    /// Vestigial auto-promote threshold retained for create-contract coverage.
    /// Current vector search derives its sign-sketch at open instead of
    /// building a persisted vote index at commit.
    #[arg(long, default_value_t = 50_000)]
    vector_auto_tier_threshold: u64,
    /// Top-k for both text and vector search.
    #[arg(long, default_value_t = 100)]
    top_k: usize,
    /// Warm-up queries before timed trials. Cycles the eval set if
    /// fewer queries are available.
    #[arg(long, default_value_t = 64)]
    warmup: usize,
    /// Measured trials; the reported p50/p95 µs is the median across
    /// trials.
    #[arg(long, default_value_t = 3)]
    trials: usize,
    /// V-curve tiers to sweep for the text bench's impact-vote-rerank
    /// calibration step.
    #[arg(long, default_value = "256,512,1024,2048,4096,8192,16384")]
    text_calibrate_grid: String,
    /// Minimum mean top-k overlap (vs Exact) the calibrate picker
    /// requires before accepting a `channel_k` tier. With the
    /// 2σ-binomial-noise margin the effective bar lands ~0.06 below
    /// this on a 100-sample run.
    #[arg(long, default_value_t = 0.90)]
    text_calibrate_target: f64,
    /// Working directory for the temp `.vls` files. Created + deleted
    /// by the bench.
    #[arg(long, default_value = "/tmp/valise-e2e-bench")]
    work_dir: PathBuf,
    /// Comma-separated list of reader-thread counts to sweep during
    /// the concurrent-search phase. Each thread holds its own
    /// `ReadConnection` and runs the warmed query set; the bench
    /// reports aggregate throughput and per-thread p50.
    #[arg(long, default_value = "1,2,4,8")]
    concurrent_readers: String,
    /// Output JSON path. Optional; if unset, only the console table is
    /// printed.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Directory for TREC run files (`qid Q0 docid rank score tag`), one
    /// per engine, scored uniformly by `bench/python/eval_runs.py`.
    #[arg(long, default_value = "bench/results/runs")]
    runs_dir: PathBuf,
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

/// Native evaluation metric of a vector dataset (what the dataset's
/// authors define neighbors under), not necessarily what Valise searches
/// with — see the module docs' "Metric handling" section.
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

/// One fully-loaded vector dataset: the first `n × dim` corpus rows and
/// `nq × dim` query rows, already L2-normalized when the dataset's
/// native metric is L2 (angular-surrogate handling; module docs).
struct VectorDataset {
    name: String,
    dir: PathBuf,
    dim: usize,
    metric: DatasetMetric,
    /// QAM/UPQ rotation block: largest power of two dividing `dim`,
    /// capped at 1024 — the production rule, which reproduces the
    /// cross-dataset experiment settings (768→256, 1536→512, 960→64,
    /// 128→128).
    block_size: usize,
    n: usize,
    nq: usize,
    corpus_len: usize,
    query_len: usize,
    /// Columns per query in the official `gt.u32`, when the dataset
    /// ships one (texmex SIFT/GIST).
    gt_k_official: Option<usize>,
    corpus: Vec<f32>,
    queries: Vec<f32>,
}

/// Exact top-`depth` neighbors (row indices, best-first) for every
/// loaded query, brute-forced under cosine over the loaded corpus
/// prefix. This is the primary recall reference for Valise *and* the
/// vector peers, so the comparison is recall-matched.
struct GroundTruth {
    depth: usize,
    /// `nq × depth`, row-major.
    ids: Vec<u32>,
}

// ---- Report shapes ---------------------------------------------------------

#[derive(Clone, Debug, serde::Serialize)]
struct Report {
    text: TextReport,
    vector: VectorReport,
    peers: PeerReport,
}

#[derive(Clone, Debug, serde::Serialize)]
struct PeerReport {
    tantivy: PeerEngine,
    usearch: PeerEngine,
    hnsw_rs: PeerEngine,
}

/// Common shape for every peer engine we benchmark in-process. Both
/// text + vector peers populate the same fields so they can sit in
/// one table.
#[derive(Clone, Debug, serde::Serialize)]
struct PeerEngine {
    name: &'static str,
    /// `"text"` or `"vector"`.
    modality: &'static str,
    corpus_size: usize,
    queries: usize,
    ingest_seconds: f64,
    /// Some engines bundle commit into ingest; in that case
    /// `commit_seconds = 0.0` and the cost is folded into `ingest`.
    commit_seconds: f64,
    storage_bytes: u64,
    storage_mib: f64,
    /// `None` for the vector engines (no analogous metric); `Some(b)`
    /// for vector peers where `b = storage / corpus_size`.
    bytes_per_vector: Option<f64>,
    p50_us: f64,
    p95_us: f64,
    /// CPU pressure during the timed query phase (None for peers we don't
    /// instrument). `effective_cores ≈ cpu_seconds / wall`.
    cpu_seconds: Option<f64>,
    effective_cores: Option<f64>,
    /// Peak process RSS (bytes) sampled via getrusage.
    peak_rss_bytes: Option<u64>,
    /// Recall vs the shared exact ground truth (vector peers only;
    /// `None` for text engines and when `--top-k` < the recall depth).
    recall_at_10: Option<f64>,
    recall_at_100: Option<f64>,
}

#[derive(Clone, Debug, serde::Serialize)]
struct TextReport {
    dataset: String,
    corpus_size: usize,
    queries_scored: usize,
    ingest_seconds: f64,
    commit_seconds: f64,
    calibrate_seconds: f64,
    chosen_channel_k: Option<usize>,
    calibration_tiers: Vec<TextCalibrationTier>,
    storage_bytes: u64,
    storage_mib: f64,
    p50_us: f64,
    p95_us: f64,
    /// CPU pressure during the timed query phase. `effective_cores ≈
    /// cpu_seconds / wall` — >1 means Valise's intra-query rayon parallelism.
    cpu_seconds: f64,
    effective_cores: f64,
    /// Peak process RSS (bytes) after the query phase, via getrusage.
    peak_rss_bytes: u64,
    concurrent: Vec<ConcurrentReport>,
}

#[derive(Clone, Debug, serde::Serialize)]
struct TextCalibrationTier {
    channel_k: usize,
    mean_overlap_at_k: f64,
    mean_latency_us: f64,
}

#[derive(Clone, Debug, serde::Serialize)]
struct VectorReport {
    dataset: String,
    corpus_size: usize,
    dim: usize,
    /// Dataset-native metric ("cosine" | "l2"), from `meta.json` or the
    /// dataset-name convention (`bench/python/valise_data.py::metric_for`).
    metric: String,
    /// What Valise actually searched with. Always "cosine": the sketch
    /// pipeline is cosine end-to-end; L2 datasets are normalized at
    /// load (angular surrogate — module docs).
    search_metric: &'static str,
    /// True when the corpus/queries were L2-normalized at load because
    /// the dataset's native metric is L2.
    l2_normalized_surrogate: bool,
    /// Codec the space was registered with ("qam" | "upq") + config.
    codec: String,
    codec_config: String,
    block_size: usize,
    /// Candidate budget used for the timed/ranked passes. `None` = the
    /// engine's production default (`max(4·k, DEFAULT_SKETCH_CANDIDATE_BUDGET)`).
    channel_k: Option<usize>,
    queries: usize,
    auto_tier_threshold: u64,
    auto_tier_fired: bool,
    ingest_seconds: f64,
    commit_seconds: f64,
    storage_bytes: u64,
    storage_mib: f64,
    bytes_per_vector: f64,
    p50_us: f64,
    p95_us: f64,
    /// Recall vs exact brute-force cosine ground truth over the loaded
    /// prefix (`None` when `--top-k` or the GT depth is below 10/100).
    recall_at_10: Option<f64>,
    recall_at_100: Option<f64>,
    /// Recall vs the official texmex `gt.u32` (L2 over the raw, full
    /// corpus). Only populated when the dataset ships one AND
    /// `--vector-n` covers the full corpus — otherwise the official GT
    /// contains neighbors outside the ingested prefix.
    recall_official_at_10: Option<f64>,
    recall_official_at_100: Option<f64>,
    /// Depth of the primary exact ground truth (≤ 100).
    gt_depth: usize,
    cpu_seconds: f64,
    effective_cores: f64,
    peak_rss_bytes: u64,
    /// Peak RSS (bytes) of a FRESH child process that only opens the
    /// committed `.vls` store and runs the warmed query set once —
    /// excludes the parent's corpus load, brute-force GT, and
    /// text-phase footprint, so it reflects what a reader actually
    /// needs (mmap + sketch + i8 cache + query overhead). `None` when
    /// the probe child failed; the bench never fails for the probe.
    peak_rss_fresh_bytes: Option<u64>,
    concurrent: Vec<ConcurrentReport>,
}

#[derive(Clone, Debug, serde::Serialize)]
struct ConcurrentReport {
    threads: usize,
    wall_seconds: f64,
    total_queries: usize,
    throughput_qps: f64,
    /// Per-thread p50 µs, averaged across the spawned readers. With
    /// no contention this should track the single-thread p50.
    mean_p50_us: f64,
    /// Speedup vs. the `threads = 1` row. > 1.0 means the readers
    /// scale; < 1.0 means contention.
    speedup_vs_single: f64,
}

// ---- Resource (CPU / memory pressure) + TREC run-file helpers ----------------

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

/// CPU work / wall time over a phase = effective cores used. ~1.0 means
/// single-threaded; >1.0 means intra-query parallelism (Valise's rayon).
fn effective_cores(cpu_seconds: f64, wall_seconds: f64) -> f64 {
    if wall_seconds > 0.0 {
        cpu_seconds / wall_seconds
    } else {
        0.0
    }
}

/// Write a standard TREC run file: `qid Q0 docid rank score tag`.
/// Evaluated uniformly (with the Python peers) by `bench/python/eval_runs.py`.
fn write_trec_run(
    runs_dir: &Path,
    engine: &str,
    dataset: &str,
    rankings: &[(String, Vec<(String, f32)>)],
) -> Result<()> {
    fs::create_dir_all(runs_dir)?;
    let path = runs_dir.join(format!("{engine}.{dataset}.run"));
    let mut out = String::new();
    for (qid, docs) in rankings {
        for (rank, (docid, score)) in docs.iter().enumerate() {
            out.push_str(&format!("{qid} Q0 {docid} {} {score} {engine}\n", rank + 1));
        }
    }
    fs::write(&path, out)?;
    println!("[runfile] {}", path.display());
    Ok(())
}

fn main() -> Result<()> {
    // Hidden fresh-process RSS probe mode: `run_vector` re-executes
    // this binary with the env var set (mirrors the self-re-exec
    // worker in `tests/crash_consistency.rs::kill_campaign`). Must be
    // checked before `Cli::parse()` — the probe uses positional args.
    if std::env::var_os(RSS_PROBE_ENV).is_some() {
        return rss_probe_child();
    }
    let cli = Cli::parse();
    fs::create_dir_all(&cli.work_dir)?;

    let text_path = cli.work_dir.join("text-only.vls");
    let vector_path = cli.work_dir.join("vector-only.vls");
    let tantivy_path = cli.work_dir.join("tantivy-index");
    let usearch_path = cli.work_dir.join("usearch.idx");
    for p in [&text_path, &vector_path, &usearch_path] {
        if p.exists() {
            fs::remove_file(p)?;
        }
    }
    if tantivy_path.exists() {
        fs::remove_dir_all(&tantivy_path)?;
    }

    println!("==== text-only Valise ====");
    let text = run_text(&cli, &text_path)?;
    print_text(&text);

    println!();
    println!("==== vector dataset + ground truth ====");
    let ds = load_vector_dataset(&cli)?;
    let gt = exact_ground_truth(&ds)?;
    let official_gt = load_official_gt(&ds)?;

    println!();
    println!("==== vector-only Valise ====");
    let vector = run_vector(&cli, &ds, &gt, official_gt.as_ref(), &vector_path)?;
    print_vector(&vector);

    println!();
    println!("==== peer engines ====");
    let peers = run_peers(&cli, &ds, &gt, &tantivy_path, &usearch_path)?;
    print_peers(&text, &vector, &peers);

    // Cleanup.
    let _ = fs::remove_file(&text_path);
    let _ = fs::remove_file(&vector_path);
    let _ = fs::remove_dir_all(&tantivy_path);
    let _ = fs::remove_file(&usearch_path);

    let report = Report {
        text,
        vector,
        peers,
    };
    if let Some(out_path) = cli.out.as_ref() {
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(out_path, serde_json::to_string_pretty(&report)?)?;
        println!();
        println!("[wrote] {:?}", out_path);
    }

    Ok(())
}

// ---- Peer engines ----------------------------------------------------------

fn run_peers(
    cli: &Cli,
    ds: &VectorDataset,
    gt: &GroundTruth,
    tantivy_dir: &Path,
    usearch_file: &Path,
) -> Result<PeerReport> {
    let corpus = read_corpus(&cli.beir_dir.join("corpus.jsonl"))?;
    let queries = read_queries(&cli.beir_dir.join("queries.jsonl"))?;
    let qrels = read_qrels(&cli.beir_dir.join("qrels/test.tsv"))?;
    let dataset = cli
        .beir_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let eval_queries: Vec<(String, String)> = queries
        .into_iter()
        .filter(|q| qrels.contains_key(&q.id))
        .map(|q| (q.id, q.text))
        .collect();
    let tantivy = run_tantivy(
        &corpus,
        &eval_queries,
        tantivy_dir,
        cli.top_k,
        cli.warmup,
        &cli.runs_dir,
        &dataset,
    )?;

    // Vector peers run on the exact corpus/queries Valise ingested (same
    // prefix, same L2-normalization for L2 datasets) and are graded
    // against the same ground truth, so recall is directly comparable.
    let usearch = run_usearch(ds, gt, cli.top_k, cli.warmup, usearch_file)?;
    let hnsw_rs = run_hnsw_rs(ds, gt, cli.top_k, cli.warmup)?;
    Ok(PeerReport {
        tantivy,
        usearch,
        hnsw_rs,
    })
}

fn run_tantivy(
    corpus: &[BeirCorpusRow],
    eval_queries: &[(String, String)],
    dir: &Path,
    top_k: usize,
    warmup: usize,
    runs_dir: &Path,
    dataset: &str,
) -> Result<PeerEngine> {
    use tantivy::collector::TopDocs;
    use tantivy::query::QueryParser;
    use tantivy::schema::{STORED, Schema, TEXT, Value};
    use tantivy::{Index, IndexWriter, TantivyDocument};

    fs::create_dir_all(dir)?;
    let mut schema_builder = Schema::builder();
    let body_field = schema_builder.add_text_field("body", TEXT | STORED);
    let id_field = schema_builder.add_text_field("doc_id", STORED);
    let schema = schema_builder.build();
    let index = Index::create_in_dir(dir, schema.clone())?;
    let mut writer: IndexWriter = index.writer(64 * 1024 * 1024)?;
    let mut payload_buf = String::new();
    let t = Instant::now();
    for row in corpus {
        payload_buf.clear();
        if !row.title.is_empty() {
            payload_buf.push_str(&row.title);
            payload_buf.push(' ');
        }
        payload_buf.push_str(&row.text);
        let mut doc = tantivy::TantivyDocument::default();
        doc.add_text(body_field, &payload_buf);
        doc.add_text(id_field, &row.id);
        writer.add_document(doc)?;
    }
    writer.commit()?;
    let ingest_seconds = t.elapsed().as_secs_f64();
    drop(writer);

    let reader = index.reader()?;
    let searcher = reader.searcher();
    let parser = QueryParser::for_index(&index, vec![body_field]);
    let parse = |q: &str| {
        parser.parse_query(q).unwrap_or_else(|_| {
            // BEIR queries occasionally contain syntax tantivy parses
            // as operators; fall back to a quoted-phrase query.
            parser
                .parse_query(&format!("\"{}\"", q.replace('"', " ")))
                .unwrap()
        })
    };
    // Warmup
    for (_, text) in eval_queries.iter().cycle().take(warmup) {
        let _ = searcher.search(&parse(text), &TopDocs::with_limit(top_k))?;
    }
    // Timed trials (CPU + peak-RSS instrumented).
    let (cpu0, _) = rusage_now();
    let wall0 = Instant::now();
    let mut samples: Vec<Duration> = Vec::with_capacity(eval_queries.len());
    for (_, text) in eval_queries {
        let parsed = parse(text);
        let t0 = Instant::now();
        let _ = searcher.search(&parsed, &TopDocs::with_limit(top_k))?;
        samples.push(t0.elapsed());
    }
    let wall = wall0.elapsed().as_secs_f64();
    let (cpu1, peak_rss) = rusage_now();
    let cpu_s = cpu1 - cpu0;

    // Ranked pass → TREC run file (retrieve the stored doc_id per hit).
    let mut rankings: Vec<(String, Vec<(String, f32)>)> = Vec::with_capacity(eval_queries.len());
    for (qid, text) in eval_queries {
        let top = searcher.search(&parse(text), &TopDocs::with_limit(top_k))?;
        let mut docs = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher.doc(addr)?;
            if let Some(id) = doc.get_first(id_field).and_then(|v| v.as_str()) {
                docs.push((id.to_string(), score));
            }
        }
        rankings.push((qid.clone(), docs));
    }
    write_trec_run(runs_dir, "tantivy", dataset, &rankings)?;

    let storage_bytes = dir_size(dir);
    Ok(PeerEngine {
        name: "tantivy",
        modality: "text",
        corpus_size: corpus.len(),
        queries: eval_queries.len(),
        ingest_seconds,
        commit_seconds: 0.0, // Tantivy bundles commit into the ingest writer
        storage_bytes,
        storage_mib: storage_bytes as f64 / (1024.0 * 1024.0),
        bytes_per_vector: None,
        p50_us: percentile_us(&samples, 50.0),
        p95_us: percentile_us(&samples, 95.0),
        cpu_seconds: Some(cpu_s),
        effective_cores: Some(effective_cores(cpu_s, wall)),
        peak_rss_bytes: Some(peak_rss),
        recall_at_10: None,
        recall_at_100: None,
    })
}

fn run_usearch(
    ds: &VectorDataset,
    gt: &GroundTruth,
    top_k: usize,
    warmup: usize,
    out_file: &Path,
) -> Result<PeerEngine> {
    use usearch::{IndexOptions, MetricKind, ScalarKind};

    let (dim, n, nq) = (ds.dim, ds.n, ds.nq);
    let opts = IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::BF16,
        connectivity: 16,
        expansion_add: 128,
        expansion_search: 64,
        multi: false,
    };
    let index = usearch::Index::new(&opts).map_err(|e| anyhow!("usearch new: {e}"))?;
    index
        .reserve(n)
        .map_err(|e| anyhow!("usearch reserve: {e}"))?;
    let t = Instant::now();
    for i in 0..n {
        let v = &ds.corpus[i * dim..(i + 1) * dim];
        index
            .add(i as u64, v)
            .map_err(|e| anyhow!("usearch add {i}: {e}"))?;
    }
    let ingest_seconds = t.elapsed().as_secs_f64();
    // usearch `save` doubles as commit: write the index to disk for
    // an apples-to-apples storage measurement.
    let t = Instant::now();
    index
        .save(&out_file.display().to_string())
        .map_err(|e| anyhow!("usearch save: {e}"))?;
    let commit_seconds = t.elapsed().as_secs_f64();

    // Warmup
    for i in 0..warmup {
        let qi = i % nq;
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let _ = index
            .search(q, top_k)
            .map_err(|e| anyhow!("usearch search: {e}"))?;
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let t0 = Instant::now();
        let _ = index
            .search(q, top_k)
            .map_err(|e| anyhow!("usearch search: {e}"))?;
        samples.push(t0.elapsed());
    }
    // Untimed ranked pass → recall vs the shared exact ground truth.
    let mut ranked: Vec<Vec<u32>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let m = index
            .search(q, top_k)
            .map_err(|e| anyhow!("usearch search: {e}"))?;
        ranked.push(m.keys.iter().map(|&k| k as u32).collect());
    }
    let recall_at_10 = recall_at(&ranked, gt, 10);
    let recall_at_100 = recall_at(&ranked, gt, 100);
    let storage_bytes = fs::metadata(out_file).map(|m| m.len()).unwrap_or(0);
    Ok(PeerEngine {
        name: "usearch",
        modality: "vector",
        corpus_size: n,
        queries: nq,
        ingest_seconds,
        commit_seconds,
        storage_bytes,
        storage_mib: storage_bytes as f64 / (1024.0 * 1024.0),
        bytes_per_vector: Some(storage_bytes as f64 / n as f64),
        p50_us: percentile_us(&samples, 50.0),
        p95_us: percentile_us(&samples, 95.0),
        cpu_seconds: None,
        effective_cores: None,
        peak_rss_bytes: None,
        recall_at_10,
        recall_at_100,
    })
}

fn run_hnsw_rs(
    ds: &VectorDataset,
    gt: &GroundTruth,
    top_k: usize,
    warmup: usize,
) -> Result<PeerEngine> {
    use hnsw_rs::prelude::{DistCosine, Hnsw};

    let (dim, n, nq) = (ds.dim, ds.n, ds.nq);
    // M = 16, ef_construction = 128 — matches the usearch settings.
    let hnsw: Hnsw<'_, f32, DistCosine> = Hnsw::new(16, n, 16, 128, DistCosine {});
    let t = Instant::now();
    // hnsw_rs supports parallel insertion; we go serial here to mirror
    // the single-thread ingest path the other engines use.
    for i in 0..n {
        let v = &ds.corpus[i * dim..(i + 1) * dim];
        hnsw.insert((v, i));
    }
    let ingest_seconds = t.elapsed().as_secs_f64();

    // hnsw_rs is an in-memory index; storage is 0 unless we explicitly
    // serialize. Skip the save-to-disk step and report 0 for storage
    // — the comparison is in latency, not on-disk footprint.
    let storage_bytes = 0u64;

    // Warmup
    for i in 0..warmup {
        let qi = i % nq;
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let _ = hnsw.search(q, top_k, 64);
    }
    let mut samples: Vec<Duration> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let t0 = Instant::now();
        let _ = hnsw.search(q, top_k, 64);
        samples.push(t0.elapsed());
    }
    // Untimed ranked pass → recall vs the shared exact ground truth.
    let mut ranked: Vec<Vec<u32>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let q = &ds.queries[qi * dim..(qi + 1) * dim];
        let hits = hnsw.search(q, top_k, 64);
        ranked.push(hits.iter().map(|h| h.d_id as u32).collect());
    }
    let recall_at_10 = recall_at(&ranked, gt, 10);
    let recall_at_100 = recall_at(&ranked, gt, 100);
    Ok(PeerEngine {
        name: "hnsw_rs",
        modality: "vector",
        corpus_size: n,
        queries: nq,
        ingest_seconds,
        commit_seconds: 0.0,
        storage_bytes,
        storage_mib: 0.0,
        bytes_per_vector: None,
        p50_us: percentile_us(&samples, 50.0),
        p95_us: percentile_us(&samples, 95.0),
        cpu_seconds: None,
        effective_cores: None,
        peak_rss_bytes: None,
        recall_at_10,
        recall_at_100,
    })
}

fn print_peers(text: &TextReport, vector: &VectorReport, peers: &PeerReport) {
    println!();
    println!("text  (BEIR {}):", text.dataset);
    println!(
        "  {:<12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>9}  {:>9}",
        "engine", "ingest s", "commit s", "size MiB", "p50 µs", "p95 µs", "speedup"
    );
    let valise_text_p50 = text.p50_us;
    let valise_text_row = (
        "Valise",
        text.ingest_seconds,
        text.commit_seconds,
        text.storage_mib,
        text.p50_us,
        text.p95_us,
    );
    print_peer_row(valise_text_row, valise_text_p50);
    print_peer_row(
        (
            peers.tantivy.name,
            peers.tantivy.ingest_seconds,
            peers.tantivy.commit_seconds,
            peers.tantivy.storage_mib,
            peers.tantivy.p50_us,
            peers.tantivy.p95_us,
        ),
        valise_text_p50,
    );

    println!();
    println!(
        "vector  ({} d={}, N={}, codec={}):",
        vector.dataset, vector.dim, vector.corpus_size, vector.codec
    );
    println!(
        "  {:<12}  {:>10}  {:>10}  {:>10}  {:>10}  {:>9}  {:>7}  {:>7}  {:>9}",
        "engine",
        "ingest s",
        "commit s",
        "size MiB",
        "p50 µs",
        "p95 µs",
        "r@10",
        "r@100",
        "speedup"
    );
    let valise_vec_p50 = vector.p50_us;
    print_vector_peer_row(
        (
            "Valise",
            vector.ingest_seconds,
            vector.commit_seconds,
            vector.storage_mib,
            vector.p50_us,
            vector.p95_us,
        ),
        vector.recall_at_10,
        vector.recall_at_100,
        valise_vec_p50,
    );
    for peer in [&peers.usearch, &peers.hnsw_rs] {
        print_vector_peer_row(
            (
                peer.name,
                peer.ingest_seconds,
                peer.commit_seconds,
                peer.storage_mib,
                peer.p50_us,
                peer.p95_us,
            ),
            peer.recall_at_10,
            peer.recall_at_100,
            valise_vec_p50,
        );
    }
}

fn print_peer_row(row: (&str, f64, f64, f64, f64, f64), valise_p50: f64) {
    let speedup = if valise_p50 > 0.0 {
        valise_p50 / row.4
    } else {
        0.0
    };
    println!(
        "  {:<12}  {:>10.3}  {:>10.3}  {:>10.2}  {:>10.0}  {:>9.0}  {:>8.2}x",
        row.0, row.1, row.2, row.3, row.4, row.5, speedup
    );
}

fn fmt_recall(r: Option<f64>) -> String {
    match r {
        Some(v) => format!("{v:.3}"),
        None => "—".to_string(),
    }
}

fn print_vector_peer_row(
    row: (&str, f64, f64, f64, f64, f64),
    r10: Option<f64>,
    r100: Option<f64>,
    valise_p50: f64,
) {
    let speedup = if valise_p50 > 0.0 {
        valise_p50 / row.4
    } else {
        0.0
    };
    println!(
        "  {:<12}  {:>10.3}  {:>10.3}  {:>10.2}  {:>10.0}  {:>9.0}  {:>7}  {:>7}  {:>8.2}x",
        row.0,
        row.1,
        row.2,
        row.3,
        row.4,
        row.5,
        fmt_recall(r10),
        fmt_recall(r100),
        speedup
    );
}

fn dir_size(path: &Path) -> u64 {
    fn walk(path: &Path) -> u64 {
        let Ok(meta) = fs::metadata(path) else {
            return 0;
        };
        if meta.is_file() {
            return meta.len();
        }
        let Ok(entries) = fs::read_dir(path) else {
            return 0;
        };
        let mut sum = 0u64;
        for entry in entries.flatten() {
            sum += walk(&entry.path());
        }
        sum
    }
    walk(path)
}

// ---- Text bench ------------------------------------------------------------

fn run_text(cli: &Cli, path: &Path) -> Result<TextReport> {
    let dataset_name = cli
        .beir_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let corpus = read_corpus(&cli.beir_dir.join("corpus.jsonl"))?;
    let queries = read_queries(&cli.beir_dir.join("queries.jsonl"))?;
    let qrels = read_qrels(&cli.beir_dir.join("qrels/test.tsv"))?;
    println!(
        "[load] {dataset_name}: corpus={} queries={} qrels_rows={}",
        corpus.len(),
        queries.len(),
        qrels.values().map(|v| v.len()).sum::<usize>(),
    );

    // ---- ingest ----
    let mut valise = ValiseFile::create(path)?;
    let collection = valise.create_collection("c")?;
    let analyzer_id = valise.register_analyzer(AnalyzerDesc {
        analyzer_id: AnalyzerId(0),
        unicode_normalization: UnicodeNormalization::Nfkc,
        case_fold: true,
        accent_fold: true,
        tokenizer: Tokenizer::UnicodeWords,
        stemming: Stemming::Porter2English,
        stopword_set_ref: None,
        stopword_query_only: true,
        stopwords: StopwordsPolicy::None,
        english_possessive_strip: true,
        min_token_len: 2,
        max_token_len: 64,
        shingle_size: 0,
        ngram_min: None,
        ngram_max: None,
        punctuation_policy: PunctuationPolicy::Drop,
    })?;
    let fs_id = valise.register_field_schema(FieldSchemaDesc {
        field_schema_id: FieldSchemaId(0),
        fields: vec![FieldDesc {
            field_id: 1,
            name: "search_text".into(),
            source: FieldSource::SearchText,
            store_positions: false,
            store_term_freq: true,
            store_set_membership: false,
            weight: 1.0,
        }],
    })?;
    let profile_id = valise.register_retrieval_profile(RetrievalProfileDesc {
        profile_id: RetrievalProfileId(0),
        profile_type: RetrievalProfileType::Bm25,
        params: RetrievalProfileParams::Bm25 {
            k1: 1.2,
            b: 0.75,
            idf_variant: IdfVariant::RobertsonSparckJones,
        },
    })?;
    let text_space_id = valise.register_text_space(TextSpaceDesc {
        text_space_id: TextSpaceId(0),
        name: "bm25".into(),
        analyzer_id,
        field_schema_id: fs_id,
        default_profile_id: profile_id,
        flags: 0,
        enabled_retrievers: 0,
    })?;

    let mut beir_id_by_frame: HashMap<u64, String> = HashMap::with_capacity(corpus.len());
    let mut payload_buf = String::new();
    let t = Instant::now();
    for row in &corpus {
        payload_buf.clear();
        if !row.title.is_empty() {
            payload_buf.push_str(&row.title);
            payload_buf.push(' ');
        }
        payload_buf.push_str(&row.text);
        let frame_id = valise.put_frame(PutFrame::document(collection, payload_buf.as_bytes()))?;
        valise.index_frame_text(frame_id, text_space_id)?;
        beir_id_by_frame.insert(frame_id.0, row.id.clone());
    }
    let ingest_seconds = t.elapsed().as_secs_f64();
    println!("[ingest] {:.2}s", ingest_seconds);

    // ---- commit ----
    let t = Instant::now();
    valise.commit()?;
    let commit_seconds = t.elapsed().as_secs_f64();
    println!("[commit] {:.2}s", commit_seconds);
    drop(valise);

    let storage_bytes = fs::metadata(path)?.len();

    // ---- reopen + calibrate + search ----
    let valise = ValiseFile::open(path, OpenMode::ReadOnly)?;
    let eval_queries: Vec<&BeirQueryRow> = queries
        .iter()
        .filter(|q| qrels.contains_key(&q.id))
        .collect();
    if eval_queries.is_empty() {
        return Err(anyhow!("text bench: no queries have qrels rows"));
    }

    let t = Instant::now();
    let grid: Vec<usize> = cli
        .text_calibrate_grid
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    let (chosen_channel_k, calibration_tiers) = calibrate_text(
        &valise,
        text_space_id,
        profile_id,
        cli.top_k,
        cli.text_calibrate_target,
        &grid,
        &eval_queries,
    )?;
    let calibrate_seconds = t.elapsed().as_secs_f64();
    println!(
        "[calibrate] {:.2}s, chose {}",
        calibrate_seconds,
        match chosen_channel_k {
            Some(k) => format!("channel_k={k}"),
            None => "Exact".to_string(),
        }
    );

    // Warmup
    for q in eval_queries.iter().cycle().take(cli.warmup) {
        let _ = valise.query_text(TextQuery {
            text_space_id,
            query: q.text.clone(),
            algorithm: QueryAlgorithm::Profile(profile_id),
            top_k: Some(cli.top_k),
            channel_k: chosen_channel_k,
        })?;
    }
    // Trials (instrumented for CPU + peak-RSS pressure).
    let (cpu0, _) = rusage_now();
    let wall0 = Instant::now();
    let (p50_us, p95_us) = time_trials(cli.trials, || {
        let mut samples: Vec<Duration> = Vec::with_capacity(eval_queries.len());
        for q in &eval_queries {
            let t0 = Instant::now();
            let _ = valise.query_text(TextQuery {
                text_space_id,
                query: q.text.clone(),
                algorithm: QueryAlgorithm::Profile(profile_id),
                top_k: Some(cli.top_k),
                channel_k: chosen_channel_k,
            })?;
            samples.push(t0.elapsed());
        }
        Ok(samples)
    })?;
    let wall = wall0.elapsed().as_secs_f64();
    let (cpu1, peak_rss_bytes) = rusage_now();
    let cpu_seconds = cpu1 - cpu0;
    let effective_cores = effective_cores(cpu_seconds, wall);

    // One ranked pass → TREC run file for uniform nDCG/MAP/Recall eval.
    let mut rankings: Vec<(String, Vec<(String, f32)>)> = Vec::with_capacity(eval_queries.len());
    for q in &eval_queries {
        let hits = valise.query_text(TextQuery {
            text_space_id,
            query: q.text.clone(),
            algorithm: QueryAlgorithm::Profile(profile_id),
            top_k: Some(cli.top_k),
            channel_k: chosen_channel_k,
        })?;
        let docs = hits
            .iter()
            .filter_map(|h| {
                beir_id_by_frame
                    .get(&h.frame_id.0)
                    .map(|id| (id.clone(), h.score))
            })
            .collect();
        rankings.push((q.id.clone(), docs));
    }
    write_trec_run(&cli.runs_dir, "valise", &dataset_name, &rankings)?;
    drop(valise);

    // ---- concurrent search ----
    let thread_counts = parse_thread_counts(&cli.concurrent_readers);
    let queries_owned: Vec<String> = eval_queries.iter().map(|q| q.text.clone()).collect();
    let queries_arc: Arc<[String]> = Arc::from(queries_owned);
    let db = Database::open(path, OpenMode::ReadOnly)?;
    let concurrent = sweep_concurrent_search(&thread_counts, |threads| {
        run_concurrent_text(
            db.clone(),
            text_space_id,
            profile_id,
            cli.top_k,
            chosen_channel_k,
            Arc::clone(&queries_arc),
            threads,
        )
    })?;
    drop(db);

    Ok(TextReport {
        dataset: dataset_name,
        corpus_size: corpus.len(),
        queries_scored: eval_queries.len(),
        ingest_seconds,
        commit_seconds,
        calibrate_seconds,
        chosen_channel_k,
        calibration_tiers,
        storage_bytes,
        storage_mib: storage_bytes as f64 / (1024.0 * 1024.0),
        p50_us,
        p95_us,
        cpu_seconds,
        effective_cores,
        peak_rss_bytes,
        concurrent,
    })
}

fn calibrate_text(
    valise: &ValiseFile,
    text_space_id: TextSpaceId,
    profile_id: RetrievalProfileId,
    top_k: usize,
    target: f64,
    grid: &[usize],
    eval_queries: &[&BeirQueryRow],
) -> Result<(Option<usize>, Vec<TextCalibrationTier>)> {
    // Sample up to 100 queries. Mirrors the paper's V-curve harness.
    let sample_n = eval_queries.len().min(100);
    let sample: Vec<&BeirQueryRow> = eval_queries.iter().take(sample_n).copied().collect();

    // Exact baseline.
    let mut exact_hits: Vec<HashSet<u64>> = Vec::with_capacity(sample.len());
    for q in &sample {
        let hits = valise.query_text(TextQuery {
            text_space_id,
            query: q.text.clone(),
            algorithm: QueryAlgorithm::Profile(profile_id),
            top_k: Some(top_k),
            channel_k: None,
        })?;
        exact_hits.push(hits.iter().map(|h| h.frame_id.0).collect());
    }

    let mut tiers: Vec<TextCalibrationTier> = Vec::with_capacity(grid.len());
    for &k in grid {
        let mut overlap_sum = 0.0;
        let mut total_lat = Duration::ZERO;
        for (q, exact) in sample.iter().zip(exact_hits.iter()) {
            let t0 = Instant::now();
            let hits = valise.query_text(TextQuery {
                text_space_id,
                query: q.text.clone(),
                algorithm: QueryAlgorithm::Profile(profile_id),
                top_k: Some(top_k),
                channel_k: Some(k),
            })?;
            total_lat += t0.elapsed();
            let approx: HashSet<u64> = hits.iter().map(|h| h.frame_id.0).collect();
            let inter = exact.intersection(&approx).count();
            overlap_sum += inter as f64 / exact.len().max(1) as f64;
        }
        let mean_overlap = overlap_sum / sample.len() as f64;
        let mean_lat_us = total_lat.as_secs_f64() * 1e6 / sample.len() as f64;
        tiers.push(TextCalibrationTier {
            channel_k: k,
            mean_overlap_at_k: mean_overlap,
            mean_latency_us: mean_lat_us,
        });
    }

    // Picker: cheapest tier whose overlap ≥ target − 2σ binomial. With
    // n=100 samples and p=0.90 the floor is ~0.84.
    let n = sample.len().max(1) as f64;
    let mut chose: Option<usize> = None;
    let mut cheapest_lat = f64::INFINITY;
    for t in &tiers {
        let p = t.mean_overlap_at_k.clamp(0.0, 1.0);
        let two_sigma = 2.0 * (p * (1.0 - p) / n).sqrt();
        let eps = (0.001_f64).max(two_sigma);
        if t.mean_overlap_at_k + eps >= target && t.mean_latency_us < cheapest_lat {
            chose = Some(t.channel_k);
            cheapest_lat = t.mean_latency_us;
        }
    }
    Ok((chose, tiers))
}

// ---- Vector bench ----------------------------------------------------------

fn run_vector(
    cli: &Cli,
    ds: &VectorDataset,
    gt: &GroundTruth,
    official_gt: Option<&OfficialGt>,
    path: &Path,
) -> Result<VectorReport> {
    let (dim, n, nq) = (ds.dim, ds.n, ds.nq);

    // ---- register codec (calibration on the first 4 096 rows) ----
    // Both families go through the production calibration-sample
    // registration API, so the bench measures the REAL end-to-end
    // path: register → ingest → commit → open (sketch derivation) →
    // sketch scan → family rerank.
    let calib_rows = n.min(4096);
    let calib_sample: Vec<Vec<f32>> = (0..calib_rows)
        .map(|i| ds.corpus[i * dim..(i + 1) * dim].to_vec())
        .collect();

    // ---- create file with auto-tier threshold tuned for this N ----
    let opts = CreateOptions {
        auto_promote: AutoPromote {
            non_f8_threshold: cli.vector_auto_tier_threshold,
            f8_threshold: cli.vector_auto_tier_threshold.saturating_mul(5),
        },
        vector: VectorContract {
            max_dim: dim as u32,
            allowed_dtypes: DtypeSet::ALL,
        },
        ..Default::default()
    };
    let mut valise = ValiseFile::create_with_options(path, opts)?;
    let collection = valise.create_collection("c")?;
    let codec_id = match cli.codec {
        // QAM (5,6): `register_codec_qam_from_sample` picks the
        // production block size (largest power of two dividing dim,
        // capped at 1024) — identical to the cross-dataset experiment
        // settings (768→256, 1536→512, 960→64, 128→128).
        CodecChoice::Qam => valise.register_codec_qam_from_sample(dim, &calib_sample)?,
        // UPQ: same block-size rule internally; cells + Empirical ring
        // design mirror `register_codec_upq_from_sample`'s defaults
        // unless `--upq-cells` overrides.
        CodecChoice::Upq => valise.register_codec_upq_from_sample_with_options(
            dim,
            cli.upq_cells,
            UpqDesignSource::Empirical,
            &calib_sample,
        )?,
    };
    let codec_config = match cli.codec {
        CodecChoice::Qam => format!(
            "qam lloyd-max (amp=5, phase=6), block_size={}, renormalize_at_decode=true",
            ds.block_size
        ),
        CodecChoice::Upq => format!(
            "upq (cells={}, design=empirical, ring_sweep=default), block_size={}",
            cli.upq_cells, ds.block_size
        ),
    };
    // The space is always registered as Cosine: Valise's sketch pipeline
    // is cosine end-to-end (module docs, "Metric handling"). L2
    // datasets were already normalized at load, which makes cosine
    // rank-equivalent to L2 on the ingested data.
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

    // ---- ingest ----
    let mut row_by_vid: HashMap<u64, u32> = HashMap::with_capacity(n);
    let t = Instant::now();
    for i in 0..n {
        let v = &ds.corpus[i * dim..(i + 1) * dim];
        let frame = valise.put_frame(PutFrame::document(collection, b""))?;
        let vid = valise.put_vector(PutVector {
            owner_frame_id: frame,
            embedding_space_id: space,
            values: v,
        })?;
        row_by_vid.insert(vid.0, i as u32);
    }
    let ingest_seconds = t.elapsed().as_secs_f64();
    println!("[ingest] {:.2}s", ingest_seconds);

    // ---- commit ----
    let t = Instant::now();
    valise.commit()?;
    let commit_seconds = t.elapsed().as_secs_f64();
    println!("[commit] {:.2}s", commit_seconds);
    drop(valise);

    let storage_bytes = fs::metadata(path)?.len();
    let auto_tier_fired = (n as u64) >= cli.vector_auto_tier_threshold;

    // ---- reopen + search ----
    // `None` lets the engine pick its production default budget
    // (`max(4·k, DEFAULT_SKETCH_CANDIDATE_BUDGET)`); the legacy `N/4`
    // point measures the codec recall ceiling instead of the shipped
    // operating point.
    let channel_k: Option<usize> = match cli.vector_channel_k {
        None => Some((n / 4).max(100)),
        Some(0) => None,
        Some(k) => Some(k),
    };
    let make_query = |q: &[f32]| VectorSearchQuery {
        embedding_space_id: space,
        query: q.to_vec(),
        k: cli.top_k,
        channel_k,
        collection_filter: None,
        fidelity: VectorFidelity::Full,
    };
    let valise = ValiseFile::open(path, OpenMode::ReadOnly)?;
    // Warmup
    for i in 0..cli.warmup {
        let qi = i % nq;
        let _ = valise.vector_search(make_query(&ds.queries[qi * dim..(qi + 1) * dim]))?;
    }
    // Trials
    let (vcpu0, _) = rusage_now();
    let vwall0 = Instant::now();
    let (p50_us, p95_us) = time_trials(cli.trials, || {
        let mut samples: Vec<Duration> = Vec::with_capacity(nq);
        for qi in 0..nq {
            let t0 = Instant::now();
            let _ = valise.vector_search(make_query(&ds.queries[qi * dim..(qi + 1) * dim]))?;
            samples.push(t0.elapsed());
        }
        Ok(samples)
    })?;
    let vwall = vwall0.elapsed().as_secs_f64();
    let (vcpu1, v_peak_rss) = rusage_now();
    let v_cpu_seconds = vcpu1 - vcpu0;
    let v_effective_cores = effective_cores(v_cpu_seconds, vwall);

    // ---- recall (untimed ranked pass, same query parameters) ----
    let mut ranked: Vec<Vec<u32>> = Vec::with_capacity(nq);
    for qi in 0..nq {
        let hits = valise.vector_search(make_query(&ds.queries[qi * dim..(qi + 1) * dim]))?;
        ranked.push(
            hits.iter()
                .filter_map(|h| row_by_vid.get(&h.vector_id.0).copied())
                .collect(),
        );
    }
    let recall_at_10 = recall_at(&ranked, gt, 10);
    let recall_at_100 = recall_at(&ranked, gt, 100);
    let (recall_official_at_10, recall_official_at_100) = match official_gt {
        Some(og) => (
            recall_at_official(&ranked, og, 10),
            recall_at_official(&ranked, og, 100),
        ),
        None => (None, None),
    };
    println!(
        "[recall] r@10={}  r@100={}  (exact {} GT over the ingested prefix, depth {})",
        fmt_recall(recall_at_10),
        fmt_recall(recall_at_100),
        if ds.metric == DatasetMetric::L2 {
            "cosine-on-normalized"
        } else {
            "cosine"
        },
        gt.depth
    );
    if official_gt.is_some() {
        println!(
            "[recall] official gt.u32 (L2, raw, full corpus): r@10={}  r@100={}",
            fmt_recall(recall_official_at_10),
            fmt_recall(recall_official_at_100),
        );
    }
    drop(valise);

    // ---- concurrent search ----
    let thread_counts = parse_thread_counts(&cli.concurrent_readers);
    let queries_arc: Arc<Vec<f32>> = Arc::new(ds.queries.clone());
    let db = Database::open(path, OpenMode::ReadOnly)?;
    let concurrent = sweep_concurrent_search(&thread_counts, |threads| {
        run_concurrent_vector(
            db.clone(),
            space,
            dim,
            cli.top_k,
            channel_k,
            nq,
            Arc::clone(&queries_arc),
            threads,
        )
    })?;
    drop(db);

    // ---- fresh-process RSS probe ----
    // The parent's getrusage peak is process-lifetime max, inflated by
    // the corpus load, brute-force GT, and the whole text phase. Re-exec
    // this binary against the still-committed store for an honest
    // reader-only measurement. Never fails the bench.
    let peak_rss_fresh_bytes = match run_rss_probe(path, ds, cli, channel_k) {
        Ok(bytes) => {
            println!(
                "[rss-probe] fresh-process reader peak RSS = {:.1} MiB (process-wide was {:.1} MiB)",
                bytes as f64 / (1024.0 * 1024.0),
                v_peak_rss as f64 / (1024.0 * 1024.0),
            );
            Some(bytes)
        }
        Err(e) => {
            eprintln!("[warn] fresh-process RSS probe failed (reporting null): {e:#}");
            None
        }
    };

    let bytes_per_vector = storage_bytes as f64 / n as f64;
    Ok(VectorReport {
        dataset: ds.name.clone(),
        corpus_size: n,
        dim,
        metric: ds.metric.as_str().to_string(),
        search_metric: "cosine",
        l2_normalized_surrogate: ds.metric == DatasetMetric::L2,
        codec: cli.codec.as_str().to_string(),
        codec_config,
        block_size: ds.block_size,
        channel_k,
        queries: nq,
        auto_tier_threshold: cli.vector_auto_tier_threshold,
        auto_tier_fired,
        ingest_seconds,
        commit_seconds,
        storage_bytes,
        storage_mib: storage_bytes as f64 / (1024.0 * 1024.0),
        bytes_per_vector,
        p50_us,
        p95_us,
        recall_at_10,
        recall_at_100,
        recall_official_at_10,
        recall_official_at_100,
        gt_depth: gt.depth,
        cpu_seconds: v_cpu_seconds,
        effective_cores: v_effective_cores,
        peak_rss_bytes: v_peak_rss,
        peak_rss_fresh_bytes,
        concurrent,
    })
}

// ---- Fresh-process RSS probe -----------------------------------------------

/// Hidden env var that switches this binary into the probe-child mode.
/// Same self-re-exec pattern as the `tests/crash_consistency.rs` kill
/// campaign (`std::env::current_exe` + env-var dispatch).
const RSS_PROBE_ENV: &str = "VALISE_E2E_RSS_PROBE";

/// Parent side: re-exec the current binary as a fresh reader against
/// the committed vector store and parse its single-JSON-line stdout
/// into a peak-RSS figure untainted by this process's history.
fn run_rss_probe(
    store: &Path,
    ds: &VectorDataset,
    cli: &Cli,
    channel_k: Option<usize>,
) -> Result<u64> {
    let exe = std::env::current_exe().context("rss probe: current_exe")?;
    let channel_k_arg = match channel_k {
        Some(k) => k.to_string(),
        None => "none".to_string(),
    };
    let out = Command::new(exe)
        .env(RSS_PROBE_ENV, "1")
        .arg(store)
        .arg(ds.dir.join("queries.f32"))
        .arg(ds.dim.to_string())
        .arg(ds.nq.to_string())
        .arg(cli.top_k.to_string())
        .arg(channel_k_arg)
        // Whether the parent L2-normalized the queries at load
        // (angular-surrogate handling for L2 datasets).
        .arg(if ds.metric == DatasetMetric::L2 {
            "1"
        } else {
            "0"
        })
        .arg(cli.warmup.to_string())
        .output()
        .context("rss probe: spawn child")?;
    if !out.status.success() {
        return Err(anyhow!(
            "rss probe child exited with {}: {}",
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
        .ok_or_else(|| anyhow!("rss probe: no JSON line on child stdout: {stdout:?}"))?;
    let v: serde_json::Value = serde_json::from_str(line).context("rss probe: parse child JSON")?;
    v.get("peak_rss_fresh_bytes")
        .and_then(serde_json::Value::as_u64)
        .ok_or_else(|| anyhow!("rss probe: missing peak_rss_fresh_bytes in {line:?}"))
}

/// Child side. Positional args (parent-controlled, no clap):
/// `<store.vls> <queries.f32> <dim> <nq> <k> <channel_k|none> <normalize 0|1> <warmup>`.
///
/// Opens the store read-only, discovers the embedding space from the
/// file (the same catalog list `vector_search` resolves the query's id
/// against; the bench registers exactly one space per dim), loads ONLY
/// the queries (nq × dim f32 — no corpus, no GT), runs warmup + one
/// full pass at `VectorFidelity::Full`, and prints its getrusage peak
/// RSS as a single JSON line. Latency is irrelevant here; RSS is the
/// target.
fn rss_probe_child() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.len() != 8 {
        return Err(anyhow!(
            "rss probe child: expected 8 args \
             (store queries_f32 dim nq k channel_k normalize warmup), got {}",
            args.len()
        ));
    }
    let store = PathBuf::from(&args[0]);
    let queries_path = PathBuf::from(&args[1]);
    let dim: usize = args[2].parse().context("rss probe child: dim")?;
    let nq: usize = args[3].parse().context("rss probe child: nq")?;
    let k: usize = args[4].parse().context("rss probe child: k")?;
    let channel_k: Option<usize> = match args[5].as_str() {
        "none" => None,
        s => Some(s.parse().context("rss probe child: channel_k")?),
    };
    let normalize = args[6] == "1";
    let warmup: usize = args[7].parse().context("rss probe child: warmup")?;
    if nq == 0 {
        return Err(anyhow!("rss probe child: nq must be ≥ 1"));
    }

    let mut queries = load_f32_prefix(&queries_path, nq * dim)?;
    if normalize {
        l2_normalize_rows(&mut queries, dim);
    }

    let valise = ValiseFile::open(&store, OpenMode::ReadOnly)?;
    let space = valise
        .embedding_spaces()
        .iter()
        .find(|s| s.dimension as usize == dim)
        .map(|s| s.embedding_space_id)
        .ok_or_else(|| {
            anyhow!("rss probe child: no embedding space with dim {dim} in {store:?}")
        })?;
    let make_query = |q: &[f32]| VectorSearchQuery {
        embedding_space_id: space,
        query: q.to_vec(),
        k,
        channel_k,
        collection_filter: None,
        fidelity: VectorFidelity::Full,
    };
    for i in 0..warmup {
        let qi = i % nq;
        let _ = valise.vector_search(make_query(&queries[qi * dim..(qi + 1) * dim]))?;
    }
    for qi in 0..nq {
        let _ = valise.vector_search(make_query(&queries[qi * dim..(qi + 1) * dim]))?;
    }
    let (_cpu, peak_rss) = rusage_now();
    println!(
        "{}",
        serde_json::json!({ "peak_rss_fresh_bytes": peak_rss })
    );
    Ok(())
}

// ---- Console output --------------------------------------------------------

fn print_text(r: &TextReport) {
    println!();
    println!("{:<22}  {}", "dataset", r.dataset);
    println!("{:<22}  {}", "corpus rows", r.corpus_size);
    println!("{:<22}  {}", "queries scored", r.queries_scored);
    println!("{:<22}  {:.3}s", "ingest", r.ingest_seconds);
    println!("{:<22}  {:.3}s", "commit", r.commit_seconds);
    println!("{:<22}  {:.3}s", "calibrate", r.calibrate_seconds);
    println!(
        "{:<22}  {}",
        "chosen channel_k",
        match r.chosen_channel_k {
            Some(k) => k.to_string(),
            None => "Exact".to_string(),
        }
    );
    for t in &r.calibration_tiers {
        println!(
            "  ck={:>5}  overlap={:.3}  mean_lat={:>6.0}µs",
            t.channel_k, t.mean_overlap_at_k, t.mean_latency_us,
        );
    }
    println!("{:<22}  {:.2} MiB", "storage", r.storage_mib);
    println!(
        "{:<22}  p50={:.0}µs  p95={:.0}µs",
        "search", r.p50_us, r.p95_us
    );
    println!(
        "{:<22}  {:.2} cores  {:.1} MiB peak RSS",
        "query pressure",
        r.effective_cores,
        r.peak_rss_bytes as f64 / (1024.0 * 1024.0)
    );
    print_concurrent(&r.concurrent);
}

fn print_vector(r: &VectorReport) {
    println!();
    println!("{:<22}  {}", "dataset", r.dataset);
    println!("{:<22}  {} (dim {})", "corpus rows", r.corpus_size, r.dim);
    println!(
        "{:<22}  {} (searched as {}{})",
        "metric",
        r.metric,
        r.search_metric,
        if r.l2_normalized_surrogate {
            ", L2-normalized at load"
        } else {
            ""
        }
    );
    println!("{:<22}  {}", "codec", r.codec_config);
    println!("{:<22}  {}", "queries", r.queries);
    println!(
        "{:<22}  threshold={}  fired={}",
        "auto-tier", r.auto_tier_threshold, r.auto_tier_fired
    );
    println!("{:<22}  {:.3}s", "ingest", r.ingest_seconds);
    println!("{:<22}  {:.3}s", "commit", r.commit_seconds);
    println!(
        "{:<22}  {:.2} MiB  ({:.1} B/vec)",
        "storage", r.storage_mib, r.bytes_per_vector
    );
    println!(
        "{:<22}  p50={:.0}µs  p95={:.0}µs",
        "search", r.p50_us, r.p95_us
    );
    println!(
        "{:<22}  r@10={}  r@100={}",
        "recall",
        fmt_recall(r.recall_at_10),
        fmt_recall(r.recall_at_100)
    );
    if r.recall_official_at_10.is_some() || r.recall_official_at_100.is_some() {
        println!(
            "{:<22}  r@10={}  r@100={}",
            "recall (official L2)",
            fmt_recall(r.recall_official_at_10),
            fmt_recall(r.recall_official_at_100)
        );
    }
    println!(
        "{:<22}  {:.2} cores  {:.1} MiB peak RSS",
        "query pressure",
        r.effective_cores,
        r.peak_rss_bytes as f64 / (1024.0 * 1024.0)
    );
    println!(
        "{:<22}  {}",
        "fresh reader RSS",
        match r.peak_rss_fresh_bytes {
            Some(b) => format!(
                "{:.1} MiB (fresh-process probe)",
                b as f64 / (1024.0 * 1024.0)
            ),
            None => "— (probe failed)".to_string(),
        }
    );
    print_concurrent(&r.concurrent);
}

fn print_concurrent(rows: &[ConcurrentReport]) {
    if rows.is_empty() {
        return;
    }
    println!("concurrent search (multiple ReadConnections, one Database):");
    println!(
        "  {:<7}  {:>8}  {:>10}  {:>10}  {:>9}",
        "threads", "wall s", "total q", "qps", "speedup"
    );
    for r in rows {
        println!(
            "  {:<7}  {:>8.3}  {:>10}  {:>10.0}  {:>8.2}x",
            r.threads, r.wall_seconds, r.total_queries, r.throughput_qps, r.speedup_vs_single,
        );
    }
}

// ---- Concurrent search helpers --------------------------------------------

fn parse_thread_counts(spec: &str) -> Vec<usize> {
    let mut out: Vec<usize> = spec
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .filter(|&n: &usize| n >= 1)
        .collect();
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        out.push(1);
    }
    out
}

fn sweep_concurrent_search(
    thread_counts: &[usize],
    mut run_one: impl FnMut(usize) -> Result<(f64, f64, usize)>,
) -> Result<Vec<ConcurrentReport>> {
    let mut out: Vec<ConcurrentReport> = Vec::with_capacity(thread_counts.len());
    let mut single_qps: Option<f64> = None;
    for &threads in thread_counts {
        let (wall_seconds, mean_p50_us, total_queries) = run_one(threads)?;
        let throughput_qps = if wall_seconds > 0.0 {
            total_queries as f64 / wall_seconds
        } else {
            0.0
        };
        if single_qps.is_none() {
            single_qps = Some(throughput_qps);
        }
        let speedup = throughput_qps / single_qps.unwrap_or(1.0);
        println!(
            "[concurrent] threads={threads}  wall={wall_seconds:.3}s  q={total_queries}  qps={throughput_qps:.0}  speedup={speedup:.2}x"
        );
        out.push(ConcurrentReport {
            threads,
            wall_seconds,
            total_queries,
            throughput_qps,
            mean_p50_us,
            speedup_vs_single: speedup,
        });
    }
    Ok(out)
}

fn run_concurrent_text(
    db: Arc<Database>,
    text_space_id: TextSpaceId,
    profile_id: RetrievalProfileId,
    top_k: usize,
    channel_k: Option<usize>,
    queries: Arc<[String]>,
    threads: usize,
) -> Result<(f64, f64, usize)> {
    let start = Instant::now();
    let mut handles = Vec::with_capacity(threads);
    for tid in 0..threads {
        let db = Arc::clone(&db);
        let queries = Arc::clone(&queries);
        handles.push(thread::spawn(move || -> Result<(usize, f64)> {
            let reader = db.reader();
            let mut samples: Vec<Duration> = Vec::with_capacity(queries.len());
            // Stagger query order per thread so caches see realistic
            // contention instead of every thread hitting the same
            // posting list at the same instant.
            let offset = tid % queries.len().max(1);
            for i in 0..queries.len() {
                let q = &queries[(offset + i) % queries.len()];
                let t0 = Instant::now();
                reader.query_text(TextQuery {
                    text_space_id,
                    query: q.clone(),
                    algorithm: QueryAlgorithm::Profile(profile_id),
                    top_k: Some(top_k),
                    channel_k,
                })?;
                samples.push(t0.elapsed());
            }
            Ok((samples.len(), percentile_us(&samples, 50.0)))
        }));
    }
    let mut total_queries = 0;
    let mut p50_sum = 0.0;
    for h in handles {
        let (n, p50) = h.join().map_err(|_| anyhow!("text worker panicked"))??;
        total_queries += n;
        p50_sum += p50;
    }
    let wall_seconds = start.elapsed().as_secs_f64();
    let mean_p50_us = p50_sum / threads as f64;
    Ok((wall_seconds, mean_p50_us, total_queries))
}

#[allow(clippy::too_many_arguments)]
fn run_concurrent_vector(
    db: Arc<Database>,
    space: valise::EmbeddingSpaceId,
    dim: usize,
    top_k: usize,
    channel_k: Option<usize>,
    nq: usize,
    queries: Arc<Vec<f32>>,
    threads: usize,
) -> Result<(f64, f64, usize)> {
    let start = Instant::now();
    let mut handles = Vec::with_capacity(threads);
    for tid in 0..threads {
        let db = Arc::clone(&db);
        let queries = Arc::clone(&queries);
        handles.push(thread::spawn(move || -> Result<(usize, f64)> {
            let reader = db.reader();
            let mut samples: Vec<Duration> = Vec::with_capacity(nq);
            let offset = tid % nq.max(1);
            for i in 0..nq {
                let qi = (offset + i) % nq;
                let q = &queries[qi * dim..(qi + 1) * dim];
                let t0 = Instant::now();
                reader.vector_search(VectorSearchQuery {
                    embedding_space_id: space,
                    query: q.to_vec(),
                    k: top_k,
                    channel_k,
                    collection_filter: None,
                    fidelity: VectorFidelity::Full,
                })?;
                samples.push(t0.elapsed());
            }
            Ok((samples.len(), percentile_us(&samples, 50.0)))
        }));
    }
    let mut total_queries = 0;
    let mut p50_sum = 0.0;
    for h in handles {
        let (n, p50) = h.join().map_err(|_| anyhow!("vector worker panicked"))??;
        total_queries += n;
        p50_sum += p50;
    }
    let wall_seconds = start.elapsed().as_secs_f64();
    let mean_p50_us = p50_sum / threads as f64;
    Ok((wall_seconds, mean_p50_us, total_queries))
}

// ---- Loaders ---------------------------------------------------------------

#[derive(Deserialize)]
struct BeirCorpusRow {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

#[derive(Deserialize)]
struct BeirQueryRow {
    #[serde(rename = "_id")]
    id: String,
    text: String,
}

#[derive(Deserialize)]
struct VectorMeta {
    dim: usize,
    corpus_len: usize,
    query_len: usize,
    /// Optional explicit metric ("cosine" | "l2"). Most meta.json files
    /// omit it; the dataset-name convention fills the gap.
    #[serde(default)]
    metric: Option<String>,
    /// Columns per query in the official `gt.u32`, when present.
    #[serde(default)]
    gt_k: Option<usize>,
}

// ---- Vector dataset loading + ground truth ---------------------------------

/// Resolve the dataset's native metric: explicit `meta.json::metric`
/// wins; otherwise the same name convention as
/// `bench/python/valise_data.py::metric_for` (cohere/openai → cosine,
/// sift/gist → l2). Unknown datasets fail loudly instead of running
/// under a silently-wrong metric.
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

/// Production rotation-block rule shared by QAM and UPQ: the largest
/// power of two dividing `dim`, capped at 1024. Reproduces the
/// cross-dataset experiment settings: 768→256, 1536→512, 960→64,
/// 128→128 (`valise_experiments/pareto/PARETO_RESEARCH_2026-06.md` Part 9).
fn block_size_for_dim(dim: usize) -> Result<usize> {
    if dim == 0 || !dim.is_multiple_of(2) {
        return Err(anyhow!(
            "dim {dim} unsupported: QAM/UPQ require a positive even dimension"
        ));
    }
    Ok((1usize << dim.trailing_zeros()).min(1024))
}

/// L2-normalize every `dim`-sized row in place (rayon).
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

fn load_vector_dataset(cli: &Cli) -> Result<VectorDataset> {
    let name = cli
        .vector_dir
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("?")
        .to_string();
    let meta_path = cli.vector_dir.join("meta.json");
    let meta: VectorMeta = serde_json::from_reader(File::open(&meta_path)?)
        .with_context(|| format!("read {meta_path:?}"))?;
    if cli.vector_n > meta.corpus_len || cli.vector_nq > meta.query_len {
        return Err(anyhow!(
            "vector bench: requested N={} nq={} exceeds dataset corpus_len={} query_len={}",
            cli.vector_n,
            cli.vector_nq,
            meta.corpus_len,
            meta.query_len
        ));
    }
    let metric = dataset_metric(&name, &meta)?;
    let block_size = block_size_for_dim(meta.dim)?;
    let dim = meta.dim;
    let mut corpus = load_f32_prefix(&cli.vector_dir.join("corpus.f32"), cli.vector_n * dim)?;
    let mut queries = load_f32_prefix(&cli.vector_dir.join("queries.f32"), cli.vector_nq * dim)?;
    if metric == DatasetMetric::L2 {
        // Angular surrogate for L2 datasets — see the module docs'
        // "Metric handling". Mirrors the cross-dataset experiments
        // (`upq_768_bench.rs::read_normalized`, PARETO Part 9), which
        // L2-normalized SIFT/GIST at load and evaluated under cosine.
        l2_normalize_rows(&mut corpus, dim);
        l2_normalize_rows(&mut queries, dim);
        println!(
            "[load] {name}: native metric is L2 — corpus + queries L2-normalized at load \
             (cosine surrogate; Valise's sketch pipeline is cosine-only)"
        );
    }
    println!(
        "[load] {name}: N={} dim={dim} nq={} metric={} block_size={block_size}",
        cli.vector_n,
        cli.vector_nq,
        metric.as_str()
    );
    Ok(VectorDataset {
        name,
        dir: cli.vector_dir.clone(),
        dim,
        metric,
        block_size,
        n: cli.vector_n,
        nq: cli.vector_nq,
        corpus_len: meta.corpus_len,
        query_len: meta.query_len,
        gt_k_official: meta.gt_k,
        corpus,
        queries,
    })
}

/// Cache directory for computed ground truth: `bench/cache/`, resolved
/// relative to the dataset directory (`bench/datasets/<name>/../../cache`)
/// so the bench works from any CWD. Same naming scheme as the Python
/// harness (`bench/python/valise_data.py`), stored as flat u32 LE
/// (`nq × depth`, row-major) instead of .npy.
fn gt_cache_path(ds: &VectorDataset, depth: usize) -> Option<PathBuf> {
    let bench_dir = ds.dir.parent()?.parent()?;
    let label = match ds.metric {
        DatasetMetric::Cosine => "cosine",
        // Distinct label: this is cosine over L2-normalized vectors,
        // NOT the Python harness's raw-L2 ground truth.
        DatasetMetric::L2 => "l2norm-cosine",
    };
    Some(bench_dir.join("cache").join(format!(
        "gt_{}_n{}_nq{}_{}_k{}.u32",
        ds.name, ds.n, ds.nq, label, depth
    )))
}

/// Exact top-`depth` neighbors under cosine over the loaded (possibly
/// normalized) corpus prefix — brute-force f32, rayon across queries,
/// cached on disk. O(nq · N · dim); ~seconds at N=100k, nq=1000, d=768.
fn exact_ground_truth(ds: &VectorDataset) -> Result<GroundTruth> {
    let depth = 100.min(ds.n);
    let cache = gt_cache_path(ds, depth);
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
    let (dim, n) = (ds.dim, ds.n);
    // Cosine = dot × inv_norm(corpus row); the query's own norm is a
    // per-query positive constant and cannot change the ranking.
    let inv_norms: Vec<f32> = ds
        .corpus
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
        "[gt] brute-force exact top-{depth} over N={} nq={} in {:.2}s",
        ds.n,
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

/// Official texmex ground truth (`gt.u32`: u32 LE, row-major, `gt_k`
/// cols per query, L2 over the RAW full corpus). Only valid when the
/// ingested prefix covers the full corpus; otherwise it references
/// neighbors the bench never ingested and is skipped with a note.
struct OfficialGt {
    gt_k: usize,
    /// `nq × gt_k` (first `nq` query rows of the file).
    ids: Vec<u32>,
}

fn load_official_gt(ds: &VectorDataset) -> Result<Option<OfficialGt>> {
    let path = ds.dir.join("gt.u32");
    if !path.exists() {
        return Ok(None);
    }
    if ds.n != ds.corpus_len {
        println!(
            "[gt] official gt.u32 present but --vector-n {} < corpus_len {} — skipping \
             (official GT is over the full corpus)",
            ds.n, ds.corpus_len
        );
        return Ok(None);
    }
    let bytes = fs::read(&path)?;
    let gt_k = ds
        .gt_k_official
        .unwrap_or_else(|| bytes.len() / 4 / ds.query_len.max(1));
    if gt_k == 0 || bytes.len() != ds.query_len * gt_k * 4 {
        return Err(anyhow!(
            "official gt.u32 size {} does not match query_len {} × gt_k {gt_k} × 4",
            bytes.len(),
            ds.query_len
        ));
    }
    let ids: Vec<u32> = bytes[..ds.nq * gt_k * 4]
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();
    println!(
        "[gt] official gt.u32 loaded (gt_k={gt_k}); note: L2 over raw vectors, while the \
         bench searches cosine over normalized vectors — reported separately"
    );
    Ok(Some(OfficialGt { gt_k, ids }))
}

/// Mean recall@k: |top-k(pred) ∩ top-k(GT)| / k over all queries.
/// `None` when the prediction depth (`--top-k`) or the GT depth can't
/// cover `k`.
fn recall_vs_rows(pred: &[Vec<u32>], gt_ids: &[u32], gt_depth: usize, k: usize) -> Option<f64> {
    if gt_depth < k || pred.is_empty() {
        return None;
    }
    if pred.iter().any(|p| p.len() < k) {
        return None;
    }
    let mut sum = 0.0f64;
    for (qi, p) in pred.iter().enumerate() {
        let truth: HashSet<u32> = gt_ids[qi * gt_depth..qi * gt_depth + k]
            .iter()
            .copied()
            .collect();
        let hits = p[..k].iter().filter(|id| truth.contains(id)).count();
        sum += hits as f64 / k as f64;
    }
    Some(sum / pred.len() as f64)
}

fn recall_at(pred: &[Vec<u32>], gt: &GroundTruth, k: usize) -> Option<f64> {
    recall_vs_rows(pred, &gt.ids, gt.depth, k)
}

fn recall_at_official(pred: &[Vec<u32>], gt: &OfficialGt, k: usize) -> Option<f64> {
    recall_vs_rows(pred, &gt.ids, gt.gt_k, k)
}

fn read_corpus(path: &Path) -> Result<Vec<BeirCorpusRow>> {
    let f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: BeirCorpusRow = serde_json::from_str(&line)
            .with_context(|| format!("{path:?}:{}: parse JSON", lineno + 1))?;
        out.push(row);
    }
    Ok(out)
}

fn read_queries(path: &Path) -> Result<Vec<BeirQueryRow>> {
    let f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let reader = BufReader::new(f);
    let mut out = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: BeirQueryRow = serde_json::from_str(&line)
            .with_context(|| format!("{path:?}:{}: parse JSON", lineno + 1))?;
        out.push(row);
    }
    Ok(out)
}

fn read_qrels(path: &Path) -> Result<HashMap<String, HashMap<String, i32>>> {
    let f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let reader = BufReader::new(f);
    let mut out: HashMap<String, HashMap<String, i32>> = HashMap::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() || line.starts_with("query-id") {
            continue;
        }
        let parts: Vec<&str> = line.split('\t').collect();
        if parts.len() != 3 {
            return Err(anyhow!(
                "{path:?}:{}: expected 3 TSV columns, got {}",
                lineno + 1,
                parts.len()
            ));
        }
        let qid = parts[0].to_string();
        let cid = parts[1].to_string();
        let score: i32 = parts[2]
            .trim()
            .parse()
            .with_context(|| format!("{path:?}:{}: non-integer relevance", lineno + 1))?;
        out.entry(qid).or_default().insert(cid, score);
    }
    Ok(out)
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

// ---- Timing utility --------------------------------------------------------

fn time_trials<F>(trials: usize, mut run_one: F) -> Result<(f64, f64)>
where
    F: FnMut() -> Result<Vec<Duration>>,
{
    let mut per_trial_p50: Vec<f64> = Vec::with_capacity(trials);
    let mut per_trial_p95: Vec<f64> = Vec::with_capacity(trials);
    for _ in 0..trials {
        let samples = run_one()?;
        per_trial_p50.push(percentile_us(&samples, 50.0));
        per_trial_p95.push(percentile_us(&samples, 95.0));
    }
    Ok((median(&mut per_trial_p50), median(&mut per_trial_p95)))
}

fn percentile_us(samples: &[Duration], p: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut us: Vec<f64> = samples.iter().map(|d| d.as_secs_f64() * 1e6).collect();
    us.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let i = ((p / 100.0) * (us.len() as f64 - 1.0)).round() as usize;
    us[i.min(us.len() - 1)]
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}
