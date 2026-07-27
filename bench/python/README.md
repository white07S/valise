# Valise third-party Python bench

Matched-recall comparison of Valise against established Python vector/text
engines, on the **same datasets and the same ground truth** used by the
Rust bench. Targets the **M4 Max, CPU-only**.

Valise itself is **not** wrapped in Python: it's measured natively by the
Rust `valise-e2e-bench` (FFI-free, fairest for Valise), and `merge_report.py`
joins its JSON (`bench/results/e2e.json`) with the Python peers' JSON on
`(dataset, modality, operating point)`. Both sides load the identical
query file and the identical cached ground truth from `datasets.py`.

## Peers (v1)

| Modality | Engines | Accuracy knob (swept for the Pareto) |
|---|---|---|
| Vector | **FAISS** Flat (GT + exact baseline), IVF-PQ, HNSW | `nprobe` / `efSearch` |
| Vector | **LanceDB** (disk IVF-PQ) | `nprobes`, `refine_factor` |
| Vector | **hnswlib** (in-memory, speed ceiling) | `ef` |
| Text | **bm25s**, **Pyserini** (Lucene), **rank_bm25** (floor) | — (exact) |

usearch + tantivy are already covered by the Rust crate bench, so they're
not re-run here. ScaNN/DiskANN are x86-only and out of scope on the M4.

## What's measured (storage + the different times are first-class)

Per the schema in `result.py`:

* **Storage** — `index_bytes`, **`bytes_per_item`** (per vector / per doc),
  and `peak_rss_bytes` (RAM-resident vs disk-resident). This is the headline
  axis for Valise's 4-bit/dim, single-file footprint vs in-memory float HNSW.
* **The different times, split** — `load` (read inputs) / `build`
  (construct index) / `persist` (write to disk) / `total_index`, *not* one
  lumped "build" number, plus query latency `p50/p95/p99/mean`, warm and
  **cold-cache** (`cold_p50_us`), and `qps`.
* **Quality** — vector: `recall@{1,10,100}` vs the shared FAISS-Flat GT;
  text: `nDCG@10 / MAP / MRR@10 / Recall@100` vs BEIR qrels.

Headline tables (per dataset): *latency & QPS at recall ≥ 0.90 / ≥ 0.95*,
with index size + build time at that operating point, plus recall-vs-QPS
Pareto plots.

## Datasets (already on disk)

* Vector: `cohere-medium-1m-f32` (d768, cosine, computed GT),
  `sift-1m` (d128, L2, official `gt.u32`), `gist-1m` (d960, L2, official GT).
* Text: `bench/beir-data/{scifact,nfcorpus,fiqa,...}` (BEIR + qrels).

## Best Valise config (what the merged Valise rows must use)

The Valise side is run through the Rust bench at its tuned config; the Python
suite is calibrated to match it (block size, metric, top-k):

* Codec: **QAM Lloyd-Max (amp=5, phase=6), block_size=256** for d768
  (`QamLloydMaxBench::with_config(dim, 256, 5, 6, true)`).
* Search: **vote-then-rerank, `fidelity = Full`**, candidate budget
  **`channel_k = N/4`** as the default operating point; the Pareto curve is
  produced by **sweeping `channel_k`** against the shared GT.
* Concurrency: lock-free pooled vote index + parallel `rerank_full` +
  adaptive fan-out admission, **`VALISE_QUERY_FANOUT=2`** (default).
* Metric per dataset matches the GT: Cohere/OpenAI → cosine (L2-normalize),
  SIFT/GIST → L2.

## Usage (uv)

```bash
cd bench/python
uv sync                      # core: numpy + faiss-cpu + psutil
uv sync --extra vector       # + lancedb, hnswlib
uv sync --extra text         # + bm25s, rank_bm25, pytrec_eval
uv sync --extra pyserini     # + pyserini (needs Java 21 on PATH)

uv run python datasets.py    # smoke test: dataset summaries + GT sanity
```

## Status

- [x] uv project + result schema (`result.py`)
- [x] dataset loaders + shared FAISS-Flat ground-truth cache (`valise_data.py`)
- [x] `harness.py` / `metrics.py` (warmup, percentiles, RSS, recall@k, pytrec_eval)
- [x] vector runners: faiss (Flat/IVF-PQ/HNSW) / hnswlib / lancedb (`run_vector.py`)
- [x] text runners: bm25s / rank_bm25 (`run_text.py`); pyserini optional (Java 21)
- [ ] cold-cache p50 (needs `sudo purge` on macOS) + concurrency sweep
- [ ] Valise `channel_k` recall-sweep hook (Rust side) + `merge_report.py` + `plots.py`

> Note: the dataset/GT module is `valise_data.py` (not `datasets.py`) — the latter
> name shadows the HuggingFace `datasets` package that LanceDB imports.
