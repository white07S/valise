// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

//! Portability benchmark: Valise (single-file hybrid store) vs a COMPOSED
//! deployment (Tantivy lexical dir + usearch vector file + JSONL payload
//! store + config) — the USENIX FAST "portability" experiment.
//!
//! Portability is measured on the axes an operator actually pays when a
//! retrieval artifact has to move between machines / be archived:
//!
//!   * **bytes + file count** — one file vs a directory of sidecars.
//!   * **packaging** — Valise is already the unit of transfer; the composed
//!     deployment must be tar-gzipped first (measured).
//!   * **open / time-to-first-query (TTFQ)** — a cold client opens the
//!     artifact and answers one text, one vector, and one hybrid query,
//!     each timed cumulatively from process entry, and emits the top-10
//!     id lists so cross-machine (macOS vs Linux) byte-identical results
//!     can be asserted.
//!   * **memory** — peak RSS of the cold probe.
//!   * **copy-under-update correctness** — copy the live artifact while a
//!     writer is mid-commit and classify every copy. Valise's footer-TOC +
//!     scan-back recovery guarantees a copy is always a valid (possibly
//!     older) snapshot with no wrong/phantom data; the composed
//!     deployment has no cross-sidecar atomicity, so a racing recursive
//!     copy can catch Tantivy and usearch at divergent counts.
//!
//! The Valise artifact is built through the PUBLIC `Store` API (text +
//! vector + payload in ONE store, one collection), exactly as an
//! application would; the composed deployment mirrors the peer-engine
//! wiring of `valise-e2e-bench` (Tantivy 0.22 `TEXT|STORED`, usearch cosine
//! BF16 M=16).
//!
//! ## Datasets (all on disk, row-aligned by document order)
//!
//! * `airgapped-manual`: text = BEIR trec-covid (171 332 docs); vectors =
//!   first 171 332 rows of `cohere-medium-1m-f32/corpus.f32` (d=768).
//! * `agent-corpus`: text = BEIR scifact (5 183 docs); vectors = first
//!   5 183 rows of the same `corpus.f32`.
//!
//! Semantic text↔vector pairing is irrelevant to every portability
//! metric here (bytes, file count, open time, copy atomicity); the
//! row-aligned pairing is disclosed for reproducibility.
//!
//! ## Subcommands
//!
//! ```text
//! build --scenario <s> --out-dir <dir>          # build BOTH deployments + build.json
//! probe --deployment <valise|composed> --path <a>  # cold client, one query per channel
//!       --queries <q.json> --out <json>
//! race  --scenario <s> --out-dir <dir>          # copy-under-update correctness
//!       --copies <N> --out <json>
//! ```
//!
//! See `bench/portability/run_remote.sh` for the cross-machine leg.

use std::{
    fs::{self, File},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};

use valise::ValiseFile;
use valise::prelude::*;

const DIM: usize = 768;
/// Collection / field names for the single Valise store.
const COLLECTION: &str = "docs";
const TEXT_FIELD: &str = "body";
const VECTOR_FIELD: &str = "dense";
/// usearch HNSW connectivity (M) — the config the paper benchmarks.
const USEARCH_M: usize = 16;
const USEARCH_EXPANSION_ADD: usize = 128;
const USEARCH_EXPANSION_SEARCH: usize = 64;

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(
    name = "valise-portability-bench",
    about = "Valise vs composed deployment portability benchmark (USENIX FAST)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build BOTH deployments from the same inputs and emit build.json.
    Build(BuildArgs),
    /// Cold-start client: open a deployment, run one query per channel.
    Probe(ProbeArgs),
    /// Copy-under-update correctness for a built deployment.
    Race(RaceArgs),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Scenario {
    #[value(name = "airgapped-manual")]
    AirgappedManual,
    #[value(name = "agent-corpus")]
    AgentCorpus,
}

impl Scenario {
    fn as_str(self) -> &'static str {
        match self {
            Scenario::AirgappedManual => "airgapped-manual",
            Scenario::AgentCorpus => "agent-corpus",
        }
    }
    /// BEIR directory (relative to `--bench-dir`) and the number of docs
    /// (== the number of leading `corpus.f32` rows) for this scenario.
    fn beir_subdir(self) -> &'static str {
        match self {
            Scenario::AirgappedManual => "beir-data/trec-covid",
            Scenario::AgentCorpus => "beir-data/scifact",
        }
    }
}

#[derive(Parser)]
struct BuildArgs {
    #[arg(long, value_enum)]
    scenario: Scenario,
    /// Output directory: `<out-dir>/valise/corpus.vls` + `<out-dir>/composed/`.
    #[arg(long)]
    out_dir: PathBuf,
    /// Root that holds `beir-data/` and `datasets/`. Defaults to `bench`
    /// (run the bench from the repo root, as `valise-e2e-bench` is).
    #[arg(long, default_value = "bench")]
    bench_dir: PathBuf,
    /// Cohere vector dataset dir (holds `corpus.f32`, `queries.f32`,
    /// `meta.json`), relative to `--bench-dir`.
    #[arg(long, default_value = "datasets/cohere-medium-1m-f32")]
    vector_subdir: PathBuf,
    /// Records per Valise commit batch (memory vs commit-count tradeoff).
    #[arg(long, default_value_t = 20_000)]
    batch: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum Deployment {
    Valise,
    Composed,
}

#[derive(Parser)]
struct ProbeArgs {
    #[arg(long, value_enum)]
    deployment: Deployment,
    /// Artifact path: the `corpus.vls` file (valise) or the `composed/` dir.
    #[arg(long)]
    path: PathBuf,
    /// `queries.json`: `{"text_query": "...", "vector_query_row": N}`.
    #[arg(long)]
    queries: PathBuf,
    /// Dataset dir holding `queries.f32` (the vector query is read from
    /// row `vector_query_row`). Must match the build's vector dataset.
    #[arg(long, default_value = "bench/datasets/cohere-medium-1m-f32")]
    dataset_dir: PathBuf,
    /// Output JSON path (also echoed to stdout).
    #[arg(long)]
    out: PathBuf,
}

#[derive(Parser)]
struct RaceArgs {
    #[arg(long, value_enum)]
    scenario: Scenario,
    /// The `--out-dir` a prior `build` wrote (holds `valise/` + `composed/`).
    #[arg(long)]
    out_dir: PathBuf,
    /// Number of racing copies taken while the writer runs.
    #[arg(long, default_value_t = 10)]
    copies: usize,
    /// Output JSON path.
    #[arg(long)]
    out: PathBuf,
    /// Deterministic seed for the copy-interval schedule.
    #[arg(long, default_value_t = 0x5EED_1234_ABCD_0001)]
    seed: u64,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.cmd {
        Command::Build(a) => cmd_build(a),
        Command::Probe(a) => cmd_probe(a),
        Command::Race(a) => cmd_race(a),
    }
}

// ---------------------------------------------------------------------------
// Shared loaders + helpers
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct BeirCorpusRow {
    #[serde(rename = "_id")]
    id: String,
    #[serde(default)]
    title: String,
    text: String,
}

/// A loaded BEIR corpus: `(doc_id, body)` in file order. `body` is
/// `title + " " + text` (matching the e2e bench's payload construction).
struct Corpus {
    docs: Vec<(String, String)>,
}

fn load_beir_corpus(path: &Path) -> Result<Corpus> {
    let f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let reader = BufReader::new(f);
    let mut docs = Vec::new();
    for (lineno, line) in reader.lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: BeirCorpusRow = serde_json::from_str(&line)
            .with_context(|| format!("{path:?}:{}: parse JSON", lineno + 1))?;
        let mut body = String::new();
        if !row.title.is_empty() {
            body.push_str(&row.title);
            body.push(' ');
        }
        body.push_str(&row.text);
        docs.push((row.id, body));
    }
    Ok(Corpus { docs })
}

/// Read the first `n_floats` little-endian f32 from `path`.
fn load_f32_prefix(path: &Path, n_floats: usize) -> Result<Vec<f32>> {
    let mut f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let need = n_floats
        .checked_mul(4)
        .ok_or_else(|| anyhow!("f32 count overflow"))?;
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

/// Read one `dim`-length query vector from `queries.f32` at `row`.
fn read_query_vector(path: &Path, row: usize, dim: usize) -> Result<Vec<f32>> {
    let mut f = File::open(path).with_context(|| format!("open {path:?}"))?;
    f.seek(SeekFrom::Start((row * dim * 4) as u64))?;
    let mut buf = vec![0u8; dim * 4];
    f.read_exact(&mut buf)
        .with_context(|| format!("read query row {row} from {path:?}"))?;
    Ok(buf
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect())
}

/// Recursive `(bytes_total, file_count)` for a file or directory.
fn tree_stats(path: &Path) -> (u64, usize) {
    let Ok(meta) = fs::metadata(path) else {
        return (0, 0);
    };
    if meta.is_file() {
        return (meta.len(), 1);
    }
    let Ok(entries) = fs::read_dir(path) else {
        return (0, 0);
    };
    let mut bytes = 0u64;
    let mut count = 0usize;
    for entry in entries.flatten() {
        let (b, c) = tree_stats(&entry.path());
        bytes += b;
        count += c;
    }
    (bytes, count)
}

/// Recursively copy `src` (file or dir) to `dst`. Best-effort per entry:
/// a racing copy of a mid-commit tree may legitimately fail to read a
/// file that is being rewritten; that is surfaced as an open error at
/// classification time, not hidden here.
fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    let meta = fs::metadata(src)?;
    if meta.is_file() {
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(src, dst)?;
        return Ok(());
    }
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        copy_tree(&entry.path(), &dst.join(entry.file_name()))?;
    }
    Ok(())
}

fn sha256_hex(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let mut f = File::open(path).with_context(|| format!("open {path:?}"))?;
    let mut hasher = Sha256::new();
    std::io::copy(&mut f, &mut hasher)?;
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(64);
    for b in digest {
        hex.push_str(&format!("{b:02x}"));
    }
    Ok(hex)
}

/// Peak process RSS in bytes via `getrusage` (macOS reports bytes, Linux
/// KiB — normalized to bytes here). Same convention as `valise-e2e-bench`.
fn peak_rss_bytes() -> u64 {
    unsafe {
        let mut ru: libc::rusage = std::mem::zeroed();
        libc::getrusage(libc::RUSAGE_SELF, &mut ru);
        let maxrss = ru.ru_maxrss as u64;
        if cfg!(target_os = "linux") {
            maxrss * 1024
        } else {
            maxrss
        }
    }
}

/// Deterministic pseudo-random unit vector seeded by `text`. Used only
/// for the race writer's fresh records (semantic content is irrelevant
/// to copy-under-update correctness). Mirrors `examples/quickstart.rs`.
fn embed(text: &str, dim: usize) -> Vec<f32> {
    let mut state = text.bytes().fold(0xcbf2_9ce4_8422_2325_u64, |h, b| {
        (h ^ b as u64).wrapping_mul(0x1000_0000_01b3)
    });
    let mut v: Vec<f32> = (0..dim)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 40) as f32 / (1u64 << 24) as f32 - 0.5
        })
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    v.iter_mut().for_each(|x| *x /= norm);
    v
}

/// A doc id as a plain string, from a db-layer [`Key`].
fn key_to_id(key: &Key) -> String {
    match key {
        Key::Str(s) => s.clone(),
        Key::U64(n) => n.to_string(),
        Key::Bytes(b) => String::from_utf8_lossy(b).into_owned(),
    }
}

// ---------------------------------------------------------------------------
// usearch options (shared by build / probe / race)
// ---------------------------------------------------------------------------

fn usearch_options(dim: usize) -> usearch::IndexOptions {
    use usearch::{IndexOptions, MetricKind, ScalarKind};
    IndexOptions {
        dimensions: dim,
        metric: MetricKind::Cos,
        quantization: ScalarKind::BF16,
        connectivity: USEARCH_M,
        expansion_add: USEARCH_EXPANSION_ADD,
        expansion_search: USEARCH_EXPANSION_SEARCH,
        multi: false,
    }
}

// ---------------------------------------------------------------------------
// Composed deployment layout + config
// ---------------------------------------------------------------------------

struct ComposedPaths {
    root: PathBuf,
    tantivy: PathBuf,
    usearch: PathBuf,
    payloads: PathBuf,
    offsets: PathBuf,
    config: PathBuf,
}

impl ComposedPaths {
    fn under(root: impl Into<PathBuf>) -> Self {
        let root = root.into();
        Self {
            tantivy: root.join("tantivy"),
            usearch: root.join("vectors.usearch"),
            payloads: root.join("payloads.jsonl"),
            offsets: root.join("payloads.offsets"),
            config: root.join("config.json"),
            root,
        }
    }
}

#[derive(Serialize, Deserialize)]
struct ComposedConfig {
    analyzer: String,
    dim: usize,
    metric: String,
    id_field: String,
    doc_count: usize,
    vector_count: usize,
}

#[derive(Serialize, Deserialize)]
struct PayloadLine {
    id: String,
    text: String,
}

// ---------------------------------------------------------------------------
// build
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct DeploymentBuild {
    path: String,
    bytes_total: u64,
    file_count: usize,
    build_seconds: f64,
    package_tar_seconds: f64,
    sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    final_commit_seconds: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tarball_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tarball_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tarball_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    note: Option<String>,
}

#[derive(Serialize)]
struct BuildReport {
    scenario: String,
    doc_count: usize,
    dim: usize,
    valise: DeploymentBuild,
    composed: DeploymentBuild,
}

fn cmd_build(a: BuildArgs) -> Result<()> {
    let beir_dir = a.bench_dir.join(a.scenario.beir_subdir());
    let vector_dir = a.bench_dir.join(&a.vector_subdir);
    let corpus_f32 = vector_dir.join("corpus.f32");

    println!(
        "[build] scenario={} out={:?}",
        a.scenario.as_str(),
        a.out_dir
    );
    println!("[load] beir corpus {:?}", beir_dir.join("corpus.jsonl"));
    let corpus = load_beir_corpus(&beir_dir.join("corpus.jsonl"))?;
    let n = corpus.docs.len();
    println!("[load] {n} docs; reading first {n} vector rows (d={DIM})");
    let vectors = load_f32_prefix(&corpus_f32, n * DIM)?;

    // Clean output.
    let valise_dir = a.out_dir.join("valise");
    let composed = ComposedPaths::under(a.out_dir.join("composed"));
    if valise_dir.exists() {
        fs::remove_dir_all(&valise_dir)?;
    }
    if composed.root.exists() {
        fs::remove_dir_all(&composed.root)?;
    }
    fs::create_dir_all(&valise_dir)?;
    fs::create_dir_all(&composed.root)?;

    // ---- Valise deployment (public Store API, one file) ----
    let valise_path = valise_dir.join("corpus.vls");
    let (valise_build_s, valise_final_commit_s) =
        build_valise(&valise_path, &corpus, &vectors, a.batch)?;
    let (valise_bytes, valise_files) = tree_stats(&valise_dir);
    let valise_sha = sha256_hex(&valise_path)?;
    println!(
        "[valise] {valise_bytes} bytes  files={valise_files}  build={valise_build_s:.2}s  \
         final_commit={valise_final_commit_s:.3}s"
    );

    // ---- Composed deployment ----
    let composed_build_s = build_composed(&composed, &corpus, &vectors)?;
    let (composed_bytes, composed_files) = tree_stats(&composed.root);

    // Packaging: tar-gzip the composed dir (the transfer unit).
    let tarball = a.out_dir.join("composed.tar.gz");
    let package_s = tar_gzip(&a.out_dir, "composed", &tarball)?;
    let tarball_bytes = fs::metadata(&tarball).map(|m| m.len()).unwrap_or(0);
    let tarball_sha = sha256_hex(&tarball)?;
    println!(
        "[composed] {composed_bytes} bytes  files={composed_files}  build={composed_build_s:.2}s  \
         tar_gzip={package_s:.2}s  tarball={tarball_bytes} bytes"
    );

    let report = BuildReport {
        scenario: a.scenario.as_str().to_string(),
        doc_count: n,
        dim: DIM,
        valise: DeploymentBuild {
            path: valise_path.display().to_string(),
            bytes_total: valise_bytes,
            file_count: valise_files,
            build_seconds: valise_build_s,
            // The Valise artifact IS the file — no packaging step. Recorded
            // as 0 per the experiment spec.
            package_tar_seconds: 0.0,
            sha256: Some(valise_sha),
            final_commit_seconds: Some(valise_final_commit_s),
            tarball_path: None,
            tarball_bytes: None,
            tarball_sha256: None,
            note: Some("single-file artifact; no packaging step (package_tar_seconds=0)".into()),
        },
        composed: DeploymentBuild {
            path: composed.root.display().to_string(),
            bytes_total: composed_bytes,
            file_count: composed_files,
            build_seconds: composed_build_s,
            package_tar_seconds: package_s,
            // sha256 of the valise file is the interop hash; for the composed
            // deployment the meaningful hash is the packaged tarball's.
            sha256: None,
            final_commit_seconds: None,
            tarball_path: Some(tarball.display().to_string()),
            tarball_bytes: Some(tarball_bytes),
            tarball_sha256: Some(tarball_sha),
            note: Some("multi-file deployment; hash + transfer the tarball".into()),
        },
    };

    let build_json = a.out_dir.join("build.json");
    fs::write(&build_json, serde_json::to_string_pretty(&report)?)?;
    println!("[wrote] {}", build_json.display());
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}

/// Build the single-file Valise store via the public `Store` API. Returns
/// `(total_build_seconds, final_commit_seconds)`.
fn build_valise(path: &Path, corpus: &Corpus, vectors: &[f32], batch: usize) -> Result<(f64, f64)> {
    let store = Store::open(path)?;
    // One collection: an English BM25 text field + a cosine vector field.
    // Codec::upq() is the family the paper benchmarks and is exposed by
    // the simple API, so it is chosen explicitly here.
    store.collection(
        COLLECTION,
        Schema::new()
            .text(TEXT_FIELD)
            .vector(VECTOR_FIELD, Vector::dim(DIM as u32).codec(Codec::upq())),
    )?;

    let batch = batch.max(1);
    let t_build = Instant::now();
    let mut final_commit_s = 0.0f64;
    let mut w = store.writer();
    for (i, (id, body)) in corpus.docs.iter().enumerate() {
        let v = &vectors[i * DIM..(i + 1) * DIM];
        w.put(
            COLLECTION,
            id.as_str(),
            Record::new().text(TEXT_FIELD, body).vector(VECTOR_FIELD, v),
        )?;
        if (i + 1) % batch == 0 {
            let t = Instant::now();
            w.commit()?;
            final_commit_s = t.elapsed().as_secs_f64();
        }
    }
    // Flush the tail (also the final commit if the last batch was partial).
    let t = Instant::now();
    let out = w.commit()?;
    if out.changed {
        final_commit_s = t.elapsed().as_secs_f64();
    }
    drop(w);
    let build_s = t_build.elapsed().as_secs_f64();
    Ok((build_s, final_commit_s))
}

/// Build the composed deployment (Tantivy index + usearch file + JSONL
/// payloads + offsets + config). Returns total build seconds.
fn build_composed(paths: &ComposedPaths, corpus: &Corpus, vectors: &[f32]) -> Result<f64> {
    use tantivy::schema::{STORED, Schema as TSchema, TEXT};
    use tantivy::{Index, IndexWriter, TantivyDocument};

    let t_build = Instant::now();
    fs::create_dir_all(&paths.tantivy)?;

    // Tantivy: body TEXT|STORED + id STORED. `row` (u64 STORED) is added
    // so the cold probe can turn a text hit into an offsets-file payload
    // fetch (row-indexed) — see the module docs / probe.
    let mut sb = TSchema::builder();
    let body_field = sb.add_text_field(TEXT_FIELD, TEXT | STORED);
    let id_field = sb.add_text_field("id", STORED);
    let row_field = sb.add_u64_field("row", STORED);
    let schema = sb.build();
    let index = Index::create_in_dir(&paths.tantivy, schema)?;
    let mut writer: IndexWriter = index.writer(64 * 1024 * 1024)?;

    // usearch: cosine, BF16, M=16.
    let opts = usearch_options(DIM);
    let uindex = usearch::Index::new(&opts).map_err(|e| anyhow!("usearch new: {e}"))?;
    uindex
        .reserve(corpus.docs.len())
        .map_err(|e| anyhow!("usearch reserve: {e}"))?;

    // Payloads (row-ordered JSONL) + per-line byte offsets.
    let mut payloads = std::io::BufWriter::new(File::create(&paths.payloads)?);
    let mut offsets = std::io::BufWriter::new(File::create(&paths.offsets)?);
    let mut byte_offset: u64 = 0;

    for (row, (id, body)) in corpus.docs.iter().enumerate() {
        // Tantivy doc.
        let mut doc = TantivyDocument::default();
        doc.add_text(body_field, body);
        doc.add_text(id_field, id);
        doc.add_u64(row_field, row as u64);
        writer.add_document(doc)?;

        // usearch vector (keyed by row).
        let v = &vectors[row * DIM..(row + 1) * DIM];
        uindex
            .add(row as u64, v)
            .map_err(|e| anyhow!("usearch add row {row}: {e}"))?;

        // Payload line + offset.
        let line = serde_json::to_string(&PayloadLine {
            id: id.clone(),
            text: body.clone(),
        })?;
        offsets.write_all(&byte_offset.to_le_bytes())?;
        payloads.write_all(line.as_bytes())?;
        payloads.write_all(b"\n")?;
        byte_offset += line.len() as u64 + 1;
    }

    writer.commit()?;
    drop(writer);
    uindex
        .save(&paths.usearch.display().to_string())
        .map_err(|e| anyhow!("usearch save: {e}"))?;
    payloads.flush()?;
    offsets.flush()?;

    let config = ComposedConfig {
        analyzer: "tantivy-default (TEXT: lowercase + simple unicode tokenizer)".into(),
        dim: DIM,
        metric: "cosine".into(),
        id_field: "id".into(),
        doc_count: corpus.docs.len(),
        vector_count: corpus.docs.len(),
    };
    fs::write(&paths.config, serde_json::to_string_pretty(&config)?)?;

    Ok(t_build.elapsed().as_secs_f64())
}

/// `tar -czf <tarball> -C <cwd> <member>`; returns wall seconds.
fn tar_gzip(cwd: &Path, member: &str, tarball: &Path) -> Result<f64> {
    let t = Instant::now();
    let status = std::process::Command::new("tar")
        .arg("-czf")
        .arg(tarball)
        .arg("-C")
        .arg(cwd)
        .arg(member)
        .status()
        .context("spawn tar")?;
    if !status.success() {
        bail!("tar failed with status {status}");
    }
    Ok(t.elapsed().as_secs_f64())
}

// ---------------------------------------------------------------------------
// probe
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct QueriesFile {
    text_query: String,
    vector_query_row: usize,
}

#[derive(Serialize)]
struct ProbeReport {
    deployment: String,
    open_ms: f64,
    first_text_hit_ms: f64,
    first_vector_hit_ms: f64,
    first_hybrid_hit_ms: f64,
    top10_text_ids: Vec<String>,
    top10_vector_ids: Vec<String>,
    top10_hybrid_ids: Vec<String>,
    peak_rss_bytes: u64,
}

fn cmd_probe(a: ProbeArgs) -> Result<()> {
    // t0 = process entry for THIS command. Nothing heavy runs before it
    // (arg parsing only); no rayon warmup, no pre-allocation.
    let t0 = Instant::now();
    let report = match a.deployment {
        Deployment::Valise => probe_valise(&a, t0)?,
        Deployment::Composed => probe_composed(&a, t0)?,
    };
    if let Some(parent) = a.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&a.out, &json)?;
    println!("{json}");
    Ok(())
}

fn probe_valise(a: &ProbeArgs, t0: Instant) -> Result<ProbeReport> {
    let q: QueriesFile = serde_json::from_reader(File::open(&a.queries)?)?;
    let qvec = read_query_vector(&a.dataset_dir.join("queries.f32"), q.vector_query_row, DIM)?;

    // Open the single-file store.
    let store = Store::open(&a.path)?;
    let open_ms = ms_since(t0);

    // Text channel: search → fetch top-1 payload (a real client needs it).
    let text_hits = store.search(
        COLLECTION,
        Search::new().text(TEXT_FIELD, &q.text_query).top_k(10),
    )?;
    let top10_text_ids: Vec<String> = text_hits.iter().map(|h| key_to_id(&h.key)).collect();
    if let Some(first) = text_hits.first() {
        let _ = store.get(COLLECTION, &first.key)?;
    }
    let first_text_hit_ms = ms_since(t0);

    // Vector channel.
    let vec_hits = store.search(
        COLLECTION,
        Search::new().vector(VECTOR_FIELD, &qvec).top_k(10),
    )?;
    let top10_vector_ids: Vec<String> = vec_hits.iter().map(|h| key_to_id(&h.key)).collect();
    if let Some(first) = vec_hits.first() {
        let _ = store.get(COLLECTION, &first.key)?;
    }
    let first_vector_hit_ms = ms_since(t0);

    // Hybrid channel: RRF fuse via the Search builder (the public path).
    let hybrid_hits = store.search(
        COLLECTION,
        Search::new()
            .text(TEXT_FIELD, &q.text_query)
            .vector(VECTOR_FIELD, &qvec)
            .fuse(Fusion::rrf(60))
            .top_k(10),
    )?;
    let top10_hybrid_ids: Vec<String> = hybrid_hits.iter().map(|h| key_to_id(&h.key)).collect();
    if let Some(first) = hybrid_hits.first() {
        let _ = store.get(COLLECTION, &first.key)?;
    }
    let first_hybrid_hit_ms = ms_since(t0);

    Ok(ProbeReport {
        deployment: "valise".into(),
        open_ms,
        first_text_hit_ms,
        first_vector_hit_ms,
        first_hybrid_hit_ms,
        top10_text_ids,
        top10_vector_ids,
        top10_hybrid_ids,
        peak_rss_bytes: peak_rss_bytes(),
    })
}

fn probe_composed(a: &ProbeArgs, t0: Instant) -> Result<ProbeReport> {
    use tantivy::collector::TopDocs;
    use tantivy::query::QueryParser;
    use tantivy::schema::Value;
    use tantivy::{Index, TantivyDocument};

    let q: QueriesFile = serde_json::from_reader(File::open(&a.queries)?)?;
    let qvec = read_query_vector(&a.dataset_dir.join("queries.f32"), q.vector_query_row, DIM)?;
    let paths = ComposedPaths::under(&a.path);

    // ---- Open the whole deployment (all sidecars) ----
    let config: ComposedConfig = serde_json::from_reader(File::open(&paths.config)?)?;
    let index = Index::open_in_dir(&paths.tantivy)?;
    let schema = index.schema();
    let body_field = schema.get_field(TEXT_FIELD)?;
    let id_field = schema.get_field("id")?;
    let row_field = schema.get_field("row")?;
    let reader = index.reader()?;
    let searcher = reader.searcher();
    let parser = QueryParser::for_index(&index, vec![body_field]);

    let uindex = usearch::Index::new(&usearch_options(config.dim))
        .map_err(|e| anyhow!("usearch new: {e}"))?;
    uindex
        .load(&paths.usearch.display().to_string())
        .map_err(|e| anyhow!("usearch load: {e}"))?;

    let offsets = load_offsets(&paths.offsets)?;
    let mut payload_file = File::open(&paths.payloads)?;
    let open_ms = ms_since(t0);

    // ---- Text channel ----
    let parse = |text: &str| {
        parser.parse_query(text).unwrap_or_else(|_| {
            parser
                .parse_query(&format!("\"{}\"", text.replace('"', " ")))
                .expect("quoted fallback query parses")
        })
    };
    let text_top = searcher.search(&parse(&q.text_query), &TopDocs::with_limit(10))?;
    let mut top10_text_ids = Vec::with_capacity(text_top.len());
    let mut text_top1_row: Option<u64> = None;
    for (rank, (_score, addr)) in text_top.iter().enumerate() {
        let doc: TantivyDocument = searcher.doc(*addr)?;
        let id = doc
            .get_first(id_field)
            .and_then(|v| v.as_str())
            .unwrap_or("?")
            .to_string();
        if rank == 0 {
            text_top1_row = doc.get_first(row_field).and_then(|v| v.as_u64());
        }
        top10_text_ids.push(id);
    }
    // One payload fetch (top-1) via the offsets file, as a real client would.
    if let Some(row) = text_top1_row {
        let _ = read_payload_at(&mut payload_file, &offsets, row as usize)?;
    }
    let first_text_hit_ms = ms_since(t0);

    // ---- Vector channel ----
    let vmatches = uindex
        .search(&qvec, 10)
        .map_err(|e| anyhow!("usearch search: {e}"))?;
    let vector_rows: Vec<u64> = vmatches.keys.clone();
    if let Some(&row) = vector_rows.first() {
        let _ = read_payload_at(&mut payload_file, &offsets, row as usize)?;
    }
    let first_vector_hit_ms = ms_since(t0);
    // Map each vector row → doc id via the payload store (for the id list).
    let mut top10_vector_ids = Vec::with_capacity(vector_rows.len());
    for &row in &vector_rows {
        let p = read_payload_at(&mut payload_file, &offsets, row as usize)?;
        top10_vector_ids.push(p.id);
    }

    // ---- Hybrid channel: RRF of the two top-10 id lists (k=60) ----
    let fused = rrf_fuse(&top10_text_ids, &top10_vector_ids, 60);
    let top10_hybrid_ids: Vec<String> = fused.iter().take(10).cloned().collect();
    // Resolve top-1 hybrid id → row for a payload fetch. Build an id→row
    // map from what the two channels already resolved.
    let mut id_to_row: std::collections::HashMap<String, u64> = std::collections::HashMap::new();
    for (id, &row) in top10_vector_ids.iter().zip(vector_rows.iter()) {
        id_to_row.insert(id.clone(), row);
    }
    for (rank, (_s, addr)) in text_top.iter().enumerate() {
        let doc: TantivyDocument = searcher.doc(*addr)?;
        if let Some(row) = doc.get_first(row_field).and_then(|v| v.as_u64()) {
            id_to_row.entry(top10_text_ids[rank].clone()).or_insert(row);
        }
    }
    if let Some(top1) = top10_hybrid_ids.first()
        && let Some(&row) = id_to_row.get(top1)
    {
        let _ = read_payload_at(&mut payload_file, &offsets, row as usize)?;
    }
    let first_hybrid_hit_ms = ms_since(t0);

    Ok(ProbeReport {
        deployment: "composed".into(),
        open_ms,
        first_text_hit_ms,
        first_vector_hit_ms,
        first_hybrid_hit_ms,
        top10_text_ids,
        top10_vector_ids,
        top10_hybrid_ids,
        peak_rss_bytes: peak_rss_bytes(),
    })
}

fn ms_since(t0: Instant) -> f64 {
    t0.elapsed().as_secs_f64() * 1e3
}

fn load_offsets(path: &Path) -> Result<Vec<u64>> {
    let bytes = fs::read(path).with_context(|| format!("read {path:?}"))?;
    if !bytes.len().is_multiple_of(8) {
        bail!(
            "offsets file {path:?} length {} not a multiple of 8",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

/// Random-access one payload line by row via the offsets index.
fn read_payload_at(file: &mut File, offsets: &[u64], row: usize) -> Result<PayloadLine> {
    let off = *offsets
        .get(row)
        .ok_or_else(|| anyhow!("payload row {row} out of range ({} offsets)", offsets.len()))?;
    file.seek(SeekFrom::Start(off))?;
    let mut reader = BufReader::new(file.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(serde_json::from_str(line.trim_end())?)
}

/// Reciprocal Rank Fusion of two ranked id lists (0-based rank `p` →
/// `1/(k + p + 1)`, summed). Deterministic: ties break by id string.
fn rrf_fuse(a: &[String], b: &[String], k: u32) -> Vec<String> {
    use std::collections::HashMap;
    let mut score: HashMap<&str, f64> = HashMap::new();
    for (p, id) in a.iter().enumerate() {
        *score.entry(id).or_insert(0.0) += 1.0 / (k as f64 + p as f64 + 1.0);
    }
    for (p, id) in b.iter().enumerate() {
        *score.entry(id).or_insert(0.0) += 1.0 / (k as f64 + p as f64 + 1.0);
    }
    let mut ranked: Vec<(&str, f64)> = score.into_iter().collect();
    ranked.sort_by(|x, y| {
        y.1.partial_cmp(&x.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| x.0.cmp(y.0))
    });
    ranked.into_iter().map(|(id, _)| id.to_string()).collect()
}

// ---------------------------------------------------------------------------
// race
// ---------------------------------------------------------------------------

/// Deterministic xorshift64* — same finalizer as `tests/crash_consistency.rs`.
struct XorShift64(u64);
impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

#[derive(Serialize)]
struct ValiseCopyOutcome {
    copy: String,
    class: String,
    served_generation: Option<u64>,
    raw_header_generation: Option<u64>,
    /// Number of race-key records present in the copy.
    race_records_present: usize,
    wrong_data: bool,
    phantom: bool,
    missing_acked: bool,
    detail: String,
}

#[derive(Serialize)]
struct ValiseRaceReport {
    deployment: String,
    scenario: String,
    copies: usize,
    committed_generations: usize,
    max_acked_generation: u64,
    counts: serde_json::Value,
    clean: bool,
    outcomes: Vec<ValiseCopyOutcome>,
}

#[derive(Serialize)]
struct ComposedCopyOutcome {
    copy: String,
    class: String,
    tantivy_docs: Option<u64>,
    usearch_size: Option<u64>,
    detail: String,
}

#[derive(Serialize)]
struct ComposedRaceReport {
    deployment: String,
    scenario: String,
    copies: usize,
    committed_docs: usize,
    counts: serde_json::Value,
    outcomes: Vec<ComposedCopyOutcome>,
}

#[derive(Serialize)]
struct RaceReport {
    scenario: String,
    copies: usize,
    valise: ValiseRaceReport,
    composed: ComposedRaceReport,
}

fn cmd_race(a: RaceArgs) -> Result<()> {
    let out_dir = &a.out_dir;
    let race_dir = out_dir.join("race");
    if race_dir.exists() {
        fs::remove_dir_all(&race_dir)?;
    }
    fs::create_dir_all(&race_dir)?;

    let valise = race_valise(&a, &race_dir)?;
    let composed = race_composed(&a, &race_dir)?;

    let report = RaceReport {
        scenario: a.scenario.as_str().to_string(),
        copies: a.copies,
        valise,
        composed,
    };
    if let Some(parent) = a.out.parent() {
        fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(&a.out, &json)?;
    println!("{json}");
    Ok(())
}

/// An acknowledged Valise write: `(key, generation, text)`.
struct Acked {
    key: String,
    generation: u64,
    text: String,
}

fn race_valise(a: &RaceArgs, race_dir: &Path) -> Result<ValiseRaceReport> {
    // Work on a COPY of the built store so the probe artifact is untouched.
    let built = a.out_dir.join("valise").join("corpus.vls");
    if !built.exists() {
        bail!("race: built Valise store not found at {built:?} (run `build` first)");
    }
    let work = race_dir.join("valise");
    fs::create_dir_all(&work)?;
    let live = work.join("corpus.vls");
    fs::copy(&built, &live)?;
    let copies_dir = work.join("copies");
    fs::create_dir_all(&copies_dir)?;

    println!(
        "[race:valise] writer loop + {} racing file copies",
        a.copies
    );

    // Copier thread: N plain file copies at seeded intervals, then stop.
    let stop = Arc::new(AtomicBool::new(false));
    let copier = {
        let stop = Arc::clone(&stop);
        let src = live.clone();
        let copies_dir = copies_dir.clone();
        let seed = a.seed;
        let n = a.copies;
        thread::spawn(move || {
            let mut rng = XorShift64::new(seed);
            let mut made = Vec::with_capacity(n);
            for i in 0..n {
                let delay = 1 + (rng.next() % 8);
                thread::sleep(Duration::from_millis(delay));
                let dst = copies_dir.join(format!("copy_{i}.vls"));
                // Plain copy of a file being rewritten by the live writer.
                let ok = fs::copy(&src, &dst).is_ok();
                made.push((dst, ok));
            }
            stop.store(true, Ordering::SeqCst);
            made
        })
    };

    // Writer loop on the main thread (append fresh key → commit → ack).
    const MAX_ITERS: usize = 20_000;
    let store = Store::open(&live)?;
    let mut acked: Vec<Acked> = Vec::new();
    {
        let mut w = store.writer();
        let mut i = 0usize;
        while !stop.load(Ordering::SeqCst) && i < MAX_ITERS {
            let key = format!("race-{i}");
            let text = format!("race payload record {i}");
            let v = embed(&key, DIM);
            w.put(
                COLLECTION,
                key.as_str(),
                Record::new()
                    .text(TEXT_FIELD, &text)
                    .vector(VECTOR_FIELD, &v),
            )?;
            let out = w.commit()?;
            acked.push(Acked {
                key,
                generation: out.snapshot_generation,
                text,
            });
            i += 1;
        }
    }
    drop(store);
    let made = copier
        .join()
        .map_err(|_| anyhow!("copier thread panicked"))?;

    let max_acked = acked.iter().map(|a| a.generation).max().unwrap_or(0);
    println!(
        "[race:valise] committed {} generations (max gen {max_acked})",
        acked.len()
    );

    // Classify + byte-verify every copy.
    let mut outcomes = Vec::with_capacity(made.len());
    let (mut valid, mut recovered, mut rejected) = (0usize, 0usize, 0usize);
    let mut clean = true;
    for (path, copied_ok) in &made {
        if !copied_ok {
            rejected += 1;
            outcomes.push(ValiseCopyOutcome {
                copy: path.display().to_string(),
                class: "rejected".into(),
                served_generation: None,
                raw_header_generation: None,
                race_records_present: 0,
                wrong_data: false,
                phantom: false,
                missing_acked: false,
                detail: "std::fs::copy failed (source mid-rewrite)".into(),
            });
            continue;
        }
        let outcome = classify_valise_copy(path, &acked, max_acked)?;
        match outcome.class.as_str() {
            "valid_snapshot" => valid += 1,
            "recovered_via_scanback" => recovered += 1,
            _ => rejected += 1,
        }
        if outcome.wrong_data || outcome.phantom || outcome.missing_acked {
            clean = false;
        }
        outcomes.push(outcome);
    }

    if !clean {
        // Loud: an Valise copy with wrong/phantom/missing acked data is a
        // potential real engine bug, not an expected composed-style race.
        eprintln!(
            "[race:valise] *** ANOMALY: at least one copy diverged from the acknowledged log ***"
        );
    }

    Ok(ValiseRaceReport {
        deployment: "valise".into(),
        scenario: a.scenario.as_str().to_string(),
        copies: a.copies,
        committed_generations: acked.len(),
        max_acked_generation: max_acked,
        counts: serde_json::json!({
            "valid_snapshot": valid,
            "recovered_via_scanback": recovered,
            "rejected": rejected,
        }),
        clean,
        outcomes,
    })
}

/// Open one Valise copy read-only and byte-verify it against the ack log.
fn classify_valise_copy(
    path: &Path,
    acked: &[Acked],
    _max_acked: u64,
) -> Result<ValiseCopyOutcome> {
    // Raw header generation (offset 72) straight from the copy bytes — the
    // fast-path anchor. If open serves a DIFFERENT generation, the engine
    // scanned back past a torn/ahead header.
    let raw_gen = read_raw_header_generation(path).ok();

    let file = match ValiseFile::open_read_only(path) {
        Ok(f) => f,
        Err(e) => {
            return Ok(ValiseCopyOutcome {
                copy: path.display().to_string(),
                class: "rejected".into(),
                served_generation: None,
                raw_header_generation: raw_gen,
                race_records_present: 0,
                wrong_data: false,
                phantom: false,
                missing_acked: false,
                detail: format!("open_read_only failed: {e}"),
            });
        }
    };
    let served = file.header().snapshot_generation;

    // Collect the race-key records actually present in the copy.
    let mut present: std::collections::HashMap<String, String> = std::collections::HashMap::new();
    let mut read_err: Option<String> = None;
    for frame in file.active_frames() {
        let uri = match file.read_frame_uri(frame.frame_id) {
            Ok(Some(u)) => u,
            Ok(None) => continue,
            Err(e) => {
                read_err = Some(format!("read_frame_uri: {e}"));
                break;
            }
        };
        let Some(Key::Str(k)) = Key::decode(&uri) else {
            continue;
        };
        if !k.starts_with("race-") {
            continue; // an original built record
        }
        match file.read_payload(frame.frame_id) {
            Ok(bytes) => {
                present.insert(k, String::from_utf8_lossy(&bytes).into_owned());
            }
            Err(e) => {
                read_err = Some(format!("read_payload({k}): {e}"));
                break;
            }
        }
    }

    // Byte-verify against the ack log.
    let mut wrong_data = false;
    let mut phantom = false;
    let mut missing_acked = false;
    let acked_keys: std::collections::HashSet<&str> =
        acked.iter().map(|a| a.key.as_str()).collect();
    // Every present race key must be acked and byte-match its acked text.
    for (k, text) in &present {
        match acked.iter().find(|a| &a.key == k) {
            None => phantom = true,
            Some(a) => {
                if &a.text != text {
                    wrong_data = true;
                }
            }
        }
    }
    // Every acked write at generation ≤ served must be present (a snapshot
    // at generation g contains every commit through g).
    for a in acked {
        if a.generation <= served && !present.contains_key(&a.key) {
            missing_acked = true;
        }
    }
    // A present race key not in the ack set is a phantom (double-checked).
    for k in present.keys() {
        if !acked_keys.contains(k.as_str()) {
            phantom = true;
        }
    }

    let class = if read_err.is_some() {
        // Opened but a committed read surfaced a typed error and no wrong
        // bytes were served — fail-safe, but not a clean snapshot.
        "rejected"
    } else if raw_gen == Some(served) {
        "valid_snapshot"
    } else {
        "recovered_via_scanback"
    };

    let detail = read_err.unwrap_or_else(|| {
        format!(
            "served_gen={served} raw_gen={:?} race_records={}",
            raw_gen,
            present.len()
        )
    });

    Ok(ValiseCopyOutcome {
        copy: path.display().to_string(),
        class: class.into(),
        served_generation: Some(served),
        raw_header_generation: raw_gen,
        race_records_present: present.len(),
        wrong_data,
        phantom,
        missing_acked,
        detail,
    })
}

/// `header.snapshot_generation` (LE u64 at byte 72) read directly from
/// the file image, before any open-time recovery.
fn read_raw_header_generation(path: &Path) -> Result<u64> {
    let mut f = File::open(path)?;
    let mut buf = [0u8; 80];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf[72..80].try_into().unwrap()))
}

fn race_composed(a: &RaceArgs, race_dir: &Path) -> Result<ComposedRaceReport> {
    use tantivy::{Index, IndexWriter, TantivyDocument};

    let built = ComposedPaths::under(a.out_dir.join("composed"));
    if !built.root.exists() {
        bail!(
            "race: built composed deployment not found at {:?} (run `build` first)",
            built.root
        );
    }
    // Work on a copy so the probe artifact is untouched.
    let work_root = race_dir.join("composed");
    copy_tree(&built.root, &work_root)?;
    let work = ComposedPaths::under(&work_root);
    let copies_dir = race_dir.join("composed-copies");
    fs::create_dir_all(&copies_dir)?;

    println!(
        "[race:composed] writer loop + {} racing recursive dir copies",
        a.copies
    );

    // Reopen the composed store for writing.
    let index = Index::open_in_dir(&work.tantivy)?;
    let schema = index.schema();
    let body_field = schema.get_field(TEXT_FIELD)?;
    let id_field = schema.get_field("id")?;
    let row_field = schema.get_field("row")?;
    let mut writer: IndexWriter = index.writer(64 * 1024 * 1024)?;

    let base_docs = index.reader()?.searcher().num_docs() as usize;
    const MAX_ITERS: usize = 20_000;
    let uindex =
        usearch::Index::new(&usearch_options(DIM)).map_err(|e| anyhow!("usearch new: {e}"))?;
    uindex
        .load(&work.usearch.display().to_string())
        .map_err(|e| anyhow!("usearch load: {e}"))?;
    uindex
        .reserve(base_docs + MAX_ITERS)
        .map_err(|e| anyhow!("usearch reserve: {e}"))?;

    // Copier thread: N recursive dir copies at seeded intervals.
    let stop = Arc::new(AtomicBool::new(false));
    let copier = {
        let stop = Arc::clone(&stop);
        let src = work.root.clone();
        let copies_dir = copies_dir.clone();
        let seed = a.seed ^ 0xABCD;
        let n = a.copies;
        thread::spawn(move || {
            let mut rng = XorShift64::new(seed);
            let mut made = Vec::with_capacity(n);
            for i in 0..n {
                let delay = 1 + (rng.next() % 8);
                thread::sleep(Duration::from_millis(delay));
                let dst = copies_dir.join(format!("copy_{i}"));
                // Recursive copy of a tree being rewritten by two engines.
                let ok = copy_tree(&src, &dst).is_ok();
                made.push((dst, ok));
            }
            stop.store(true, Ordering::SeqCst);
            made
        })
    };

    // Writer loop: add doc → tantivy commit → usearch add → usearch save.
    let mut committed = base_docs;
    let mut i = 0usize;
    while !stop.load(Ordering::SeqCst) && i < MAX_ITERS {
        let row = base_docs + i;
        let key = format!("race-{i}");
        let text = format!("race payload record {i}");
        let v = embed(&key, DIM);
        let mut doc = TantivyDocument::default();
        doc.add_text(body_field, &text);
        doc.add_text(id_field, &key);
        doc.add_u64(row_field, row as u64);
        writer.add_document(doc)?;
        writer.commit()?;
        uindex
            .add(row as u64, &v)
            .map_err(|e| anyhow!("usearch add: {e}"))?;
        uindex
            .save(&work.usearch.display().to_string())
            .map_err(|e| anyhow!("usearch save: {e}"))?;
        committed = row + 1;
        i += 1;
    }
    drop(writer);
    let made = copier
        .join()
        .map_err(|_| anyhow!("copier thread panicked"))?;
    println!("[race:composed] committed up to {committed} docs");

    // Classify each copy: do BOTH components open, and do their counts agree?
    let mut outcomes = Vec::with_capacity(made.len());
    let (mut ok_consistent, mut ok_inconsistent, mut open_error, mut partial) =
        (0usize, 0usize, 0usize, 0usize);
    for (dir, copied_ok) in &made {
        let paths = ComposedPaths::under(dir);
        let tantivy_docs: Option<u64> = if !copied_ok {
            None
        } else {
            Index::open_in_dir(&paths.tantivy)
                .and_then(|idx| Ok(idx.reader()?.searcher().num_docs()))
                .ok()
        };
        let usearch_size: Option<u64> = if !copied_ok {
            None
        } else {
            usearch::Index::new(&usearch_options(DIM))
                .and_then(|idx| {
                    idx.load(&paths.usearch.display().to_string())?;
                    Ok(idx.size() as u64)
                })
                .ok()
        };
        let (class, detail) = match (tantivy_docs, usearch_size) {
            (Some(t), Some(u)) if t == u => {
                ok_consistent += 1;
                ("ok_consistent", format!("both open; counts agree ({t})"))
            }
            (Some(t), Some(u)) => {
                ok_inconsistent += 1;
                (
                    "ok_inconsistent",
                    format!("both open but tantivy={t} != usearch={u} (silent divergence)"),
                )
            }
            (None, None) => {
                open_error += 1;
                ("open_error", "neither component opened".to_string())
            }
            (t, u) => {
                partial += 1;
                (
                    "partial",
                    format!("one component unusable (tantivy={t:?} usearch={u:?})"),
                )
            }
        };
        outcomes.push(ComposedCopyOutcome {
            copy: dir.display().to_string(),
            class: class.into(),
            tantivy_docs,
            usearch_size,
            detail,
        });
    }

    Ok(ComposedRaceReport {
        deployment: "composed".into(),
        scenario: a.scenario.as_str().to_string(),
        copies: a.copies,
        committed_docs: committed,
        counts: serde_json::json!({
            "ok_consistent": ok_consistent,
            "ok_inconsistent": ok_inconsistent,
            "open_error": open_error,
            "partial": partial,
        }),
        outcomes,
    })
}
