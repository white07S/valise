# Reproducing the Valise end-to-end benchmark

> Looking for the durability claims instead? Crash consistency has its own
> harnesses and its own document: **[CRASH_CAMPAIGN.md](CRASH_CAMPAIGN.md)**.

One Rust binary — `valise-e2e-bench` — builds two `.vls` files from real
corpora, measures every lifecycle phase, then runs the same workloads
through three peer engines in the same process for head-to-head
comparison.

| Phase       | Text (BM25)                       | Vector (`--codec qam` \| `upq`)            |
|-------------|------------------------------------|-------------------------------------------|
| ingest      | `put_frame` + `index_frame_text`   | `put_frame` + `put_vector`                |
| commit      | flush text segments, fsync         | flush codec bytes (no vector build)       |
| calibrate   | V-curve over `channel_k` tiers     | (n/a)                                     |
| search      | `query_text` (BM25 + impact-vote)  | `vector_search` (sign-sketch scan + family rerank, ck=N/4) |
| recall      | (nDCG via TREC run files, §8)      | recall@10 / recall@100 vs exact GT for Valise **and** the vector peers |
| concurrent  | N reader threads × eval queries    | N reader threads × eval queries           |
| storage     | `.vls` size on disk                | same                                       |
| peers       | **Tantivy** inline (BM25)          | **usearch** + **hnsw_rs** inline (HNSW)   |

A single JSON report lands at `--out`. Working `.vls` files are
deleted at the end of the run. The bench also records **CPU pressure**
(`effective_cores ≈ cpu_seconds / wall` over the query phase) and **peak
RSS** per modality, and writes TREC run files (`--runs-dir`) so retrieval
quality (nDCG@10/MAP/Recall) is scored uniformly — see §8.

For head-to-head quality + CPU/memory against **Python** engines
(FAISS, LanceDB, hnswlib · bm25s, pyserini, rank_bm25) on the *same*
datasets and ground truth, see **§8 (`bench/python/`)**.

## 0. Get the data

The corpora are large and regenerable, so they are not in the repository.
Fetch and convert them with:

```bash
python3 bench/prep_data.py --list        # what's available, and how big
python3 bench/prep_data.py all-small     # scifact + sift-1m, ~504 MB
```

`scifact` needs nothing but the standard library. The vector datasets
ship as HDF5, so they need `h5py` and `numpy` (`pip install h5py numpy`).
Downloads resume-safely — a partial transfer is never mistaken for a
finished one — and anything already present is skipped unless you pass
`--force`.

Two datasets cannot be fetched automatically: `cohere-medium-1m-f32` is
behind HuggingFace's gated-terms flow, and `openai-medium-500k` is not
redistributable. `prep_data.py --list` explains how to build each by hand,
and `--layout` prints the exact on-disk format expected.

## 1. Prerequisites

- macOS aarch64 (M-series) or Linux x86_64 with the stable Rust
  toolchain pinned in `rust-toolchain.toml`.
- **BEIR scifact** (5 183 docs, 300 evaluated queries) at
  `bench/beir-data/scifact/` — `prep_data.py scifact` puts it there:
  ```
  bench/beir-data/scifact/
    corpus.jsonl
    queries.jsonl
    qrels/test.tsv
  ```
- A vector dataset at `bench/datasets/<name>/`:
  ```
  bench/datasets/<name>/
    corpus.f32          # row-major little-endian f32, corpus_len × dim
    queries.f32         # row-major little-endian f32, query_len × dim
    meta.json           # {"dim": …, "corpus_len": …, "query_len": …}
    gt.u32              # optional official GT (texmex; u32 LE, gt_k cols/query)
  ```
  The dimension is read from `meta.json` — the bench is no longer
  pinned to d=768. Supported datasets (see the matrix in §5):

  | dataset                  | dim  | native metric | block | codecs    | how it runs                              |
  |--------------------------|------|---------------|-------|-----------|-------------------------------------------|
  | cohere-medium-1m-f32     |  768 | cosine        |  256  | qam, upq  | as-is                                     |
  | openai-medium-500k       | 1536 | cosine        |  512  | qam, upq  | as-is                                     |
  | sift-1m                  |  128 | L2            |  128  | qam, upq  | L2-normalized at load, cosine surrogate   |
  | gist-1m                  |  960 | L2            |   64  | qam, upq  | L2-normalized at load, cosine surrogate   |

  The rotation block size is the production rule (largest power of two
  dividing `dim`, capped at 1024). The metric comes from `meta.json`'s
  optional `"metric"` field or the dataset-name convention
  (`bench/python/valise_data.py::metric_for`); unknown datasets fail with
  a clear error instead of running under a silently-wrong metric.

## 2. Build

```bash
cargo build --release -p valise-bench --bin valise-e2e-bench
```

## 3. Run

```bash
target/release/valise-e2e-bench \
    --beir-dir bench/beir-data/scifact \
    --vector-dir bench/datasets/sift-1m \
    --vector-n 100000 \
    --out bench/results/e2e.json
```

That is the fully reproducible configuration — both datasets come from
`prep_data.py all-small`. Swap `--vector-dir` for
`bench/datasets/cohere-medium-1m-f32` to match the d=768 reference numbers
in §6, which is the corpus most of this document was measured on.

The run takes a couple of minutes on an Apple M-series and writes
`bench/results/e2e.json`. No persistent `.vls` artefacts remain on disk
afterward.

**Sanity check.** On an M-series laptop the command above should land near:

| Metric | Valise | Peer |
|---|---|---|
| text storage | 5.8 MiB | Tantivy 7.9 MiB |
| text p50 | ~129 µs | Tantivy ~192 µs |
| vector recall@10 | 0.933 | usearch 0.991, hnsw_rs 0.979 |
| vector storage | 16.1 MiB | usearch 38.6 MiB |
| concurrent text scaling | ~8.7x at 8 threads | — |

Vector p50 (~812 µs) is slower than the HNSW peers by design: there is no
persisted graph, so the sketch scan is linear in the corpus. The trade is
storage and the absence of an index build. Treat these as order-of-magnitude
guides — absolute numbers move with hardware, dimension, and distribution.

## 4. CLI knobs

| Flag                            | Default       | Meaning                                                                                              |
|---------------------------------|---------------|------------------------------------------------------------------------------------------------------|
| `--beir-dir`                    | _required_    | BEIR dataset directory for the text file.                                                            |
| `--vector-dir`                  | _required_    | Cohere-style dataset directory for the vector file (corpus.f32 + queries.f32 + meta.json).           |
| `--vector-n`                    | `100_000`     | First N corpus rows ingested into the vector Valise.                                                    |
| `--vector-nq`                   | `1_000`       | Queries timed during the vector search trials.                                                       |
| `--codec`                       | `qam`         | Vector codec for the embedding space: `qam` = QAM Lloyd-Max (5,6); `upq` = UPQ (Empirical ring design). Both calibrate on the first 4 096 rows and search through the normal `vector_search` path. |
| `--upq-cells`                   | `2048`        | UPQ cell budget (2048 = 11 bits/pair = 5.5 bits/dim, the storage-equivalent of QAM (5,6)).           |
| `--vector-channel-k`            | unset         | Vector candidate budget. Unset = legacy `N/4` (recall-ceiling mode); `0` = the engine's production default (`max(4·k, 2048)`); any other value verbatim. Recorded as `channel_k` in the JSON (`null` = production default). |
| `--vector-auto-tier-threshold`  | `50_000`      | Vote-index `auto_promote_non_f8`. Tuned so the build fires for the default `--vector-n`.             |
| `--top-k`                       | `100`         | Top-K for both text and vector search.                                                               |
| `--warmup`                      | `64`          | Warm-up queries before the timed trials.                                                             |
| `--trials`                      | `3`           | Trials; final p50/p95 is the median across trials.                                                   |
| `--text-calibrate-grid`         | `256..16384`  | Comma-separated `channel_k` tiers for the V-curve sweep.                                             |
| `--text-calibrate-target`       | `0.90`        | Target mean top-k overlap vs Exact. Picker adds a 2σ-binomial-noise margin (~0.06 at n=100 samples). |
| `--work-dir`                    | `/tmp/...`    | Where the temp `.vls` files land. Created + deleted by the bench.                                    |
| `--concurrent-readers`          | `1,2,4,8`     | Reader-thread counts to sweep during the concurrent phase. Each thread holds its own `ReadConnection` against a shared `Database`. |
| `--out`                         | unset         | Optional JSON output path.                                                                           |
| `--runs-dir`                    | `bench/results/runs` | Directory for TREC run files (`qid Q0 docid rank score tag`), one per engine, scored by `bench/python/eval_runs.py`. |

### Environment knobs

| Env var             | Default | Meaning                                                                                                                     |
|---------------------|---------|-----------------------------------------------------------------------------------------------------------------------------|
| `VALISE_QUERY_FANOUT`  | `2`     | Max queries allowed to fan out across the Rayon pool at once; the excess run serially. `0` = unlimited (always parallel). Caps oversubscription under concurrent readers — see §5. |

## 5. What's measured

### Text-only Valise

1. Register an analyzer (UnicodeWords, NFKC, case-fold, accent-fold,
   Porter-2 stemming, possessive strip), a single-field BM25 retrieval
   profile (`k1=1.2`, `b=0.75`, `RobertsonSparckJones` IDF), and a
   text space binding them.
2. Ingest every BEIR document via `put_frame` + `index_frame_text`.
3. `commit()` — text-index segments flush, fsync.
4. **Calibrate**: sample 100 queries-with-qrels, run an Exact baseline
   (`channel_k=None`), then sweep each grid tier; pick the cheapest
   tier whose mean top-100 overlap with Exact ≥ `target − 2σ`.
5. Warm-up + N trials of `query_text` with the chosen `channel_k`.

### Vector-only Valise

1. Calibrate the selected codec on the first 4 096 rows of the corpus
   — `--codec qam` uses `register_codec_qam_from_sample` (QAM (5,6)
   Lloyd-Max), `--codec upq` uses
   `register_codec_upq_from_sample_with_options` (`--upq-cells`,
   Empirical ring design). Both pick the production block size
   (largest power of two dividing dim, capped at 1024: 768→256,
   1536→512, 960→64, 128→128).
2. `register_embedding_space` (primary codec only; no secondary).
3. Ingest every vector via `put_vector` (`put_frame` with empty
   payload).
4. `commit()` — the codec byte stream flushes. No vector-side index is
   built at commit; the **sign-sketch** index (1 bit/dim, derived for
   free from the stored codes) is built in-memory at file-open.
5. Warm-up + N trials of `vector_search` with the `--vector-channel-k`
   budget (default: legacy `N/4`, min 100; `0` = production default)
   and `fidelity = Full` (sign-sketch Hamming scan →
   family rerank — QAM-sliding i8 or UPQ decoded-i8 — → f32/exact
   rerank oversample).
6. An untimed ranked pass computes **recall@10 / recall@100** with the
   same query parameters (fields `recall_at_10` / `recall_at_100` in
   the JSON, also reported for usearch + hnsw_rs so the comparison is
   recall-matched).

### Metric handling + ground truth

Valise's sketch pipeline is **cosine end-to-end**: stage 1 is an angular
sign-sketch Hamming scan, the QAM (5,6) sliding stage-2 kernel scores
`-dot·inv_norm` regardless of `space.metric`, and the UPQ rerank path
is hard-coded cosine. Datasets whose native metric is **L2**
(SIFT/GIST) are therefore run the way the cross-dataset experiments
did (PARETO_RESEARCH_2026-06.md Part 9): every vector is
**L2-normalized at load** and searched under cosine — on unit vectors
cosine and L2 rank identically, so this is an *angular surrogate* for
the L2 task, not native L2. The report records this in
`metric` (native), `search_metric` (always `"cosine"`) and
`l2_normalized_surrogate`.

Ground truth for `recall_at_*` is **exact brute-force top-100 under
cosine over the loaded (possibly normalized) corpus prefix**, computed
with rayon and cached at
`bench/cache/gt_<name>_n<N>_nq<NQ>_<metric>_k100.u32` (flat u32 LE,
same naming scheme as `bench/python/valise_data.py`; the `l2norm-cosine`
label marks surrogate GT as distinct from the Python harness's raw-L2
GT). The official texmex `gt.u32` (L2 over the **raw, full** corpus)
is only comparable when `--vector-n` covers the full corpus; in that
case recall against it is additionally reported as
`recall_official_at_10` / `recall_official_at_100`, and it is skipped
with a console note otherwise.

### Concurrent search (both modalities)

After the single-thread phase finishes, the bench opens one
`Arc<Database>` (read-only) and sweeps `--concurrent-readers` thread
counts. Each thread calls `db.reader()` to acquire its own pinned
`ReadConnection`, then runs the full eval-query set against it. The
bench reports per-thread-count:

- **wall s**       — wall-clock of the whole sweep across all threads
- **total q**      — sum of queries served
- **qps**          — `total_q / wall_s`
- **speedup**      — `qps(N) / qps(1)`

Scaling behaviour we expect:

- **Text** scales near-linearly (≈ 0.9–1.0× per thread up to physical
  core count). BM25 is mostly hot-cache; the shared `bm25_cache` uses
  `parking_lot::Mutex` only on cache *misses*, which are rare after
  warm-up.
- **Vector** is CPU-bound at `channel_k = N/4`: one query already fans
  out across every P-core (vote accumulate + the rerank over `channel_k`
  candidates), so absolute speedup tops out around **2.4×** on a 12-P-core
  machine no matter how many reader threads you add. Two pieces keep that
  ceiling clean rather than pathological:
  - The vote index is **lock-free** — `Arc<VoteIndex>` (immutable) with
    per-query scratch drawn from an internal pool, so readers no longer
    serialize on a `Mutex<VoteIndex>`.
  - An **adaptive fan-out limiter** (`VALISE_QUERY_FANOUT`, default 2) caps
    how many queries fan out across the shared Rayon pool at once; the
    excess run **serially** (one core each) instead of oversubscribing the
    pool. This trades a higher p50 under heavy load for a much tighter p95
    and higher throughput. Set `VALISE_QUERY_FANOUT=0` to disable (always
    parallel) — lower p50 under load, but the p95 tail balloons.

The concurrent phase also indirectly exercises the snapshot pinning
machinery — three integration tests (`tests/concurrent_search.rs`)
prove the same hot path produces deterministic top-k results under
contention and stays stable while a `WriteConnection` commits.

## 6. Reference numbers

From a fresh run on an Apple M-series, release build, single process,
defaults (`scifact` + `cohere 100k`):

| Subject     | ingest s | commit s | storage MiB | B/vec | p50 µs | p95 µs | cores¹ | peak RSS |
|-------------|---------:|---------:|------------:|------:|-------:|-------:|-------:|---------:|
| text Valise    |    0.34  |    0.11  |       5.80  |  —    |   124  |   213  |  1.00  |  229 MiB |
| vector Valise  |    0.15  |    0.46  |      58.85  | 617.1 |  1061  |  1294  |  1.00  | 1141 MiB |

¹ `effective_cores = cpu_seconds / wall` over the query phase: text BM25
is single-threaded; vector sketch-scan + QAM-sliding rerank runs
single-threaded too at this size (the per-query work fits cleanly in
one core). Multi-thread throughput scales with reader count — see the
concurrency sweep below.

### Concurrent search reference numbers

Text (scifact, 300 queries × N threads):

| threads | wall s | total q | qps     | speedup |
|--------:|-------:|--------:|--------:|--------:|
|       1 |  0.048 |     300 |   6,241 |   1.00× |
|       2 |  0.042 |     600 |  14,223 |   2.28× |
|       4 |  0.045 |   1,200 |  26,444 |   4.24× |
|       8 |  0.046 |   2,400 |  51,959 |   8.33× |

Vector (Cohere 100k, 1 000 queries × N threads, sketch-scan + QAM-sliding
rerank). Per-thread p50/p95 are averaged across the reader threads.
Measured via `bench/examples/vote_trace.rs`:

| threads | qps   | speedup | p50 µs | p95 µs |
|--------:|------:|--------:|-------:|-------:|
|       1 |   730 |   1.00× |  1,049 |  1,267 |
|       2 | 1,497 |   2.05× |  1,309 |  1,785 |
|       4 | 1,815 |   2.49× |  1,425 |  3,690 |
|       8 | 2,305 |   3.16× |  2,992 |  4,058 |

### Linux x86_64 (AVX2) reference numbers

Lab box: Intel Core i7-8550U (Kaby Lake R, AVX2 + FMA + POPCNT, no
AVX-VNNI, no AVX-512), Debian 13. Built with the project default
`-C target-cpu=x86-64-v3` (set in `.cargo/config.toml`), which
statically enables AVX2 + FMA + POPCNT + BMI2 — the dispatchers
collapse to direct AVX2 kernel calls at compile time. Numbers below
are from the Phase-6 close-out of the AVX2 SIMD rollout
(see `docs/VECTOR_SEARCH.md` for the kernel design).

| Subject     | ingest s | commit s | storage MiB | B/vec | p50 µs | p95 µs | cores | peak RSS |
|-------------|---------:|---------:|------------:|------:|-------:|-------:|------:|---------:|
| text Valise    |   1.018  |   0.445  |       5.80  |  —    |   316  |   555  |  1.00 |  209 MiB |
| vector Valise  |   0.836  |   1.269  |      58.85  | 617.1 |  6,722 |  6,987 |  6.96 |  757 MiB |

Vector qps sweep (Phase 6, lab box):

| threads | qps   | speedup vs 1-thread |
|--------:|------:|--------------------:|
|       1 |   132 |              1.00×  |
|       2 |   152 |              1.16×  |
|       4 |   156 |              1.18×  |
|       8 |   161 |              1.23×  |

Lab vs M-series, side-by-side:

| metric | M-series | lab AVX2 | lab/M |
|---|---:|---:|---:|
| text p50 µs    |   124 |   316 | 2.55× |
| vector p50 µs  | 1,061 | 6,722 | 6.34× |
| vector qps@1   |   730 |   132 | 5.53× |
| storage MiB    | 58.85 | 58.85 | 1.00× |
| B/vec          | 617.1 | 617.1 | 1.00× |

### Peer engines on lab (head-to-head with Valise AVX2)

Same `valise-e2e-bench` run. Both arches use the same Cargo command —
the peers are not rebuilt with Valise-specific flags.

Text — BEIR scifact (5,183 docs, 300 queries):

| engine    | ingest s | commit s | size MiB | p50 µs | p95 µs | speedup vs Valise |
|-----------|---------:|---------:|---------:|-------:|-------:|---------------:|
| **Valise**   |    1.02  |   0.45   |    5.80  |    316 |    555 |    **1.00×**   |
| tantivy   |    0.32  |   0.00   |    7.90  |    459 |    907 |        0.69×   |

Vector — Cohere d=768, 100,000 rows (1,000 queries):

| engine    | ingest s | commit s | size MiB | B/vec  | p50 µs | p95 µs | speedup vs Valise |
|-----------|---------:|---------:|---------:|-------:|-------:|-------:|---------------:|
| **Valise**   |    0.84  |   1.27   |   58.85  |  617.1 |  6,722 |  6,987 |    **1.00×**   |
| usearch   |   78.05  |   0.13   |  160.64  | 1,675  |    699 |    837 |        9.61×   |
| hnsw_rs   |  339.27  |   0.00   |   _mem_  |   —    |  2,847 |  3,465 |        2.36×   |

Notes:

- **Vector e2e regression vs M-series (6.34×) > the average kernel
  regression (~2-3×).** Root cause: at the default `channel_k =
  N/4 = 25,000`, the QAM-sliding rerank dominates the per-query
  budget (≥ 75% of wall), and the rerank IS the `raw_dot_int`
  kernel — which lands at **5.8× slower than NEON SDOT** on AVX2
  (the §4.3 ISA limit: no SDOT-equivalent, no 64-byte register-
  resident TBL4). At smaller `channel_k`, the lab/M ratio collapses
  toward the average kernel ratio.
- **Storage is byte-identical to the M-series build** (58.85 MiB at
  d=768 N=100k). Format invariance under arch change is enforced by
  `tests/golden_format_v2.rs`.
- **Text path is essentially unchanged** — no SIMD hot loop in BM25
  scoring; the lab/M ratio matches the underlying CPU's IPC ratio.
- **qps speedup vs 1-thread plateaus at 1.23× on 8 threads** — the
  i7-8550U has 4 physical cores (8 SMT threads), and the rerank's
  `mullo_epi32` chain saturates the integer pipes once two threads
  share a physical core. M-series equivalent reaches 3.16× at 8
  threads thanks to higher core count and integer throughput.

## 7. Build recipes for arch-specific SIMD

The QAM hot kernels in `src/codec/qam_lloyd_max/simd/` and
`src/retrieval/sketch/` have **two dispatch paths**:

- **Static dispatch** — when the build is told a target_feature is
  available at compile time (e.g. via `-C target-feature=+avx2,+fma`),
  the dispatchers collapse to direct kernel calls. No runtime CPU-
  feature check, and the AVX2 kernel can be inlined into the caller
  since they share the target_feature context (§4.2 structural rule).
- **Runtime dispatch** — when no target_feature is set, the
  dispatchers run `is_x86_feature_detected!()` (one cached atomic
  load per call) and route to the AVX2 kernel if available, else the
  scalar fallback. Works on any x86_64 chip, including pre-Haswell.

### Recipes

```sh
# === Production (default): static AVX2 baseline ===
# The repo ships `.cargo/config.toml` with `-C target-cpu=x86-64-v3`
# for x86_64 targets. v3 = Haswell (2013+) = AVX2 + FMA + POPCNT +
# BMI2 + LZCNT. Statically dispatched on AVX2 chips; will not run on
# pre-Haswell Intel or pre-Excavator AMD.
cargo build --release

# === Maximum performance (host-specific): native CPU ===
# Uses every feature available on the build host. The fastest possible
# binary, but won't run on any older or different CPU. Use for
# benchmarking, never for shipping.
cargo build-native             # alias defined in .cargo/config.toml
# equivalent to:
RUSTFLAGS="-C target-cpu=native" cargo build --release

# === Portable (legacy x86_64 compatibility): runtime dispatch ===
# Drops `target-cpu`, so the build is compatible with any x86_64
# (Westmere 2010 and later). Dispatchers do the runtime CPU check.
# Slower on AVX2-capable chips because the call boundary isn't
# inlinable, but works everywhere.
cargo build-portable           # alias defined in .cargo/config.toml

# === aarch64 (no flags needed) ===
# NEON + dotprod (SDOT) are part of the v8.2-a baseline that Apple
# Silicon ships. The dispatchers always emit the NEON path.
cargo build --release
```

### Why the static / runtime split exists

- The benches use the static path so kernel timings reflect best-
  case ISA dispatch (`bench/baseline/avx2_phase{2..6}_i7_8550u.json`).
- The portable path exists for distributing pre-built artifacts that
  must run on any x86_64 (e.g. Docker images for unknown hosts).
- The runtime check is one `OnceLock<bool>` atomic load per kernel
  entry — negligible vs the kernel work, but it does prevent the
  AVX2 kernel from being inlined into the dispatcher.

### Verifying which path your build took

```sh
# Quick disasm check: with static dispatch, the dispatcher inlines
# the AVX2 body and you'll see `vfmadd*ps` directly. With runtime
# dispatch, you'll see a `call` instruction to the AVX2 entry point.
cargo asm --release --lib valise::codec::qam_lloyd_max::simd::asymmetric_dot_pairs \
  | grep -E "vfmadd|call|callq" | head -10
```

Notes:
- Vector commit does not build any index — the sign-sketch is derived
  in-memory from the QAM phase codes at file-open.
- Vector storage = full `.vls` (header + QAM bytes + catalog + TOC
  footer). 617 B/vec is the **bytes-per-vector** under QAM (5, 6); the
  sketch lives only in RAM.
- Text calibrate on scifact picks `channel_k=256` because the corpus
  is small enough that the smallest tier already hits overlap ≥ 0.99.
- p50/p95 are median across `--trials`; per-query latencies are not
  averaged.

### Peer engines (head-to-head, same machine, same JSON output)

Text — BEIR scifact (5 183 docs, 300 queries):

| engine    | ingest s | commit s | size MiB | p50 µs | p95 µs | speedup vs Valise |
|-----------|---------:|---------:|---------:|-------:|-------:|---------------:|
| **Valise**   |    0.36  |   0.10   |    5.80  |    126 |    224 |    **1.00×**   |
| tantivy   |    0.23  |   0.00   |    7.90  |    200 |    358 |        0.63×   |

Vector — Cohere `text-embedding-3-large` d=768, first 100k rows
(1 000 queries):

| engine    | ingest s | commit s | size MiB | B/vec  | p50 µs | p95 µs | speedup vs Valise |
|-----------|---------:|---------:|---------:|-------:|-------:|-------:|---------------:|
| **Valise**   |    0.15  |   0.46   |   58.85  |  617.1 |  1,061 |  1,294 |    **1.00×**   |
| usearch   |   38.35  |   0.13   |  160.68  | 1685.0 |    337 |    458 |        3.15×   |
| hnsw_rs   |  120.03  |   0.00   |   _mem_  |   _—_  |  1,079 |  1,346 |        0.98×   |

(`_mem_` = hnsw_rs is in-memory only; no disk artefact.
`speedup vs Valise = valise_p50 / engine_p50`, so > 1× means the peer is
faster at the cost of a heavier build and/or larger footprint.)

Trade-off picture:

- **Text.** Valise's BM25 wins on p50, p95, ingest **and quality**:
  scored by one `pytrec_eval` (§8), Valise's nDCG@10 = **0.6829** vs
  tantivy's 0.6522 — and it even edges Lucene/pyserini's 0.6789.
- **Vector.** Valise wins build time (256× faster than usearch, 800×
  faster than hnsw_rs), storage (2.7× smaller than usearch), and
  recall (0.97 vs the vote pipeline's 0.90, also reached single-thread
  now). usearch still wins per-query latency (~3× faster on one core),
  but hnsw_rs is now ~at-parity (0.98×). Valise's trade: a small
  per-query O(N) sketch scan in exchange for ~free build, ~3× smaller
  on-disk footprint, and no graph/tuning surface.

## 8. Cleanup invariants

The bench is idempotent: it removes any pre-existing `text-only.vls`
or `vector-only.vls` in `--work-dir` before starting, and deletes
both at the end of a successful run. The only file left behind is
the JSON at `--out` (when supplied).

No data caches (`bench/data*/`, `bench/logs/`, `bench/results/*`
older than this run) are created or read, with one exception: the
exact-ground-truth cache under `bench/cache/` (see §5, "Metric
handling + ground truth") is written on first run per
`(dataset, N, nq)` and reused afterwards. Delete it freely — it is
recomputed on demand.

## 9. Third-party Python peers (`bench/python/`)

Compares Valise against established Python engines on the **same datasets
and the same ground truth**: vector recall vs an exact FAISS-Flat GT,
text nDCG vs BEIR qrels. Valise is measured natively (Rust, FFI-free); the
Rust bench's TREC run files (`--runs-dir`) are scored by the *same*
`pytrec_eval` the Python engines use, so quality is directly comparable.
`uv` provisions the env (Python 3.11 auto-installed).

```bash
cd bench/python
uv sync --extra vector --extra text      # FAISS, LanceDB, hnswlib, bm25s, rank_bm25
uv sync --extra pyserini                 # optional; needs Homebrew openjdk@21

# vector recall/latency Pareto (sweeps each engine's accuracy knob)
uv run python run_vector.py --dataset cohere-medium-1m-f32 --n 100000 --nq 1000

# text quality + latency (bm25s/rank_bm25; add pyserini if synced)
JAVA_HOME="$(brew --prefix openjdk@21)/libexec/openjdk.jdk/Contents/Home" \
  uv run python run_text.py --dataset scifact --engines bm25s,rank_bm25,pyserini

# score the Rust bench's Valise + tantivy run files with the same evaluator
uv run python eval_runs.py scifact \
  ../results/runs/valise.scifact.run ../results/runs/tantivy.scifact.run
```

> The dataset/GT module is `valise_data.py`, **not** `datasets.py` — the
> latter shadows the HuggingFace `datasets` package LanceDB imports.

### Reference — text (scifact, 300 queries, top-k=100; all nDCG via pytrec_eval)

| stack | engine | nDCG@10 | MAP | R@100 | p50 µs | cores | B/doc | RSS MiB |
|-------|--------|--------:|----:|------:|-------:|------:|------:|--------:|
| Rust   | **Valise** (BM25)   | **0.6829** | 0.643 | 0.912 |  124 | 1.00 | 1174 |  229 |
| Python | pyserini (Lucene)| 0.6789 | 0.640 | 0.925 |  947 | 5.2¹ |  219 | 1110 |
| Python | bm25s            | 0.6617 | 0.626 | 0.876 |   72 | 1.28 | 1006 |  108 |
| Rust   | tantivy          | 0.6522 | 0.609 | 0.876 |  191 | 1.00 | 1599 |  n/a² |
| Python | rank_bm25        | 0.5670 | 0.527 | 0.791 | 5339 | 1.00 |  —   |  206 |

¹ JVM GC/JIT background threads during the loop, not search parallelism.
² Rust bench runs all phases in one process → tantivy's getrusage maxrss
is process-cumulative, not tantivy-specific.

### Reference — vector (Cohere-768, N=100k) at matched recall@100 ≈ 0.90

| stack | engine | recall@100 | p50 µs | cores | B/vec | RSS MiB | build s |
|-------|--------|-----------:|-------:|------:|------:|--------:|--------:|
| Rust   | **Valise** (ck=25k)    | 0.904 | 1182 | 9.96 |  746 | 1141 |  0.51 |
| Python | faiss-hnsw (ef=64)  | 0.884 |  243 | 1.00 | 3344 | 1801 | 49.8  |
| Python | faiss-hnsw (ef=128) | 0.953 |  420 | 1.00 | 3344 | 1801 | 49.8  |
| Python | hnswlib (ef=16)     | 0.885 |  714 | 1.00 | 3221 | 1817 | 130   |
| Python | faiss-ivfpq (max)   | 0.632 |  435 | 1.00 |  151 | 1801 | 24    |
| Python | lancedb (max)       | 0.602 | 10995| 2.55 | 3231 | 3418 | 28    |

Takeaways: Valise is the **storage + build-time + memory champion at high
recall** — 4.3× smaller index than the HNSW peers, ~100–260× faster
build, and the lowest peak RSS (mmap/disk-first) — while spending ~10
cores/query to keep wall-latency competitive. The HNSW peers win raw
single-core latency; the PQ engines (faiss-ivfpq, lancedb) are tiny/fast
but cap at ~0.6 recall. RSS is not perfectly cross-stack (Rust vs Python
process baselines differ); the effective-cores numbers are clean
per-phase deltas.
