# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Unified result schema for every benchmarked engine (vector + text).

Two emphases baked in per the bench plan:

* **Storage** is first-class — `index_bytes` + `bytes_per_item`, plus the
  in-memory `peak_rss_bytes`, so a disk-first engine (Valise, LanceDB) is
  compared honestly against an in-memory one (hnswlib holds every float
  in RAM and writes nothing useful to disk).
* **The different times are split**, not lumped into one "build" number:
  `load` (read inputs) / `build` (construct index) / `persist` (write to
  disk) — because engines spend time very differently (LanceDB writes as
  it builds; hnswlib builds in RAM then dumps; Valise's commit is a separate
  fsync'd phase).

`to_dict()` also emits the Rust `PeerEngine` aliases (`ingest_seconds`,
`commit_seconds`, `storage_bytes`, `p50_us`, `p95_us`) so
`merge_report.py` can join these rows with `bench/results/e2e.json`.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class EngineResult:
    # --- identity ---
    engine: str  # "faiss-ivfpq", "lancedb", "hnswlib", "bm25s", "pyserini", "valise", ...
    modality: str  # "vector" | "text"
    dataset: str  # "cohere-medium-1m-f32", "scifact", ...
    n: int  # corpus rows indexed
    nq: int  # queries timed
    top_k: int
    dim: int | None = None  # vector only
    metric: str | None = None  # "cosine" | "l2" (vector only)

    # --- operating point on the recall/latency Pareto ---
    # The accuracy knob this row was measured at (FAISS nprobe, HNSW ef,
    # LanceDB nprobes, Valise channel_k, ...). One EngineResult == one point.
    knob_name: str | None = None
    knob_value: float | None = None

    # --- times (seconds), split by phase ---
    load_seconds: float | None = None  # read corpus into the engine's input form
    build_seconds: float | None = None  # construct the index structure
    persist_seconds: float | None = None  # flush/commit/write the index to disk
    total_index_seconds: float | None = (
        None  # load + build + persist (ingest+commit analog)
    )

    # --- storage / footprint ---
    index_bytes: int | None = None  # on-disk index size
    bytes_per_item: float | None = None  # index_bytes / n  (per vector or per doc)
    peak_rss_bytes: int | None = None  # process RSS high-water during query
    # What `index_bytes` counts, for the paper's storage-accounting
    # disclosure (e.g. "codes+ivf_lists+ids", "graph+f32_vectors+ids",
    # "sqlite_pages_whole_file").
    bytes_include: str | None = None

    # --- query latency (microseconds), warm cache, single thread ---
    p50_us: float | None = None
    p95_us: float | None = None
    p99_us: float | None = None
    mean_us: float | None = None
    qps: float | None = None  # single-thread throughput
    qps_threads: int | None = None  # threads used if a concurrent run
    cold_p50_us: float | None = None  # first-query latency after dropping page cache

    # --- CPU / memory pressure ---
    cpu_seconds: float | None = None  # CPU time consumed during the query phase
    effective_cores: float | None = None  # cpu_seconds / wall (~1.0 = single-core)
    rss_mean_bytes: int | None = None  # mean process RSS over build+query

    # --- quality ---
    # vector
    recall_at_1: float | None = None
    recall_at_10: float | None = None
    recall_at_100: float | None = None
    # text (vs BEIR qrels)
    ndcg_at_10: float | None = None
    map_score: float | None = None
    mrr_at_10: float | None = None
    recall_at_100_text: float | None = None

    # --- provenance ---
    lib_version: str | None = None
    threads: int | None = None
    notes: str | None = None
    env: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.env:
            self.env = default_env()

    @property
    def bpv(self) -> float | None:
        """bytes per item (vector/doc), derived from current index_bytes."""
        return (self.index_bytes / self.n) if (self.index_bytes and self.n) else None

    def to_dict(self) -> dict:
        d = asdict(self)
        # Derived footprint (index_bytes is set after construction).
        d["bytes_per_item"] = self.bpv
        # Rust PeerEngine aliases for merge_report.py.
        d["ingest_seconds"] = self.load_seconds or self.build_seconds
        d["commit_seconds"] = self.persist_seconds
        d["storage_bytes"] = self.index_bytes
        d["storage_mib"] = (
            (self.index_bytes / (1024 * 1024)) if self.index_bytes else None
        )
        return d


def default_env() -> dict:
    return {
        "machine": platform.machine(),
        "system": platform.system(),
        "processor": platform.processor(),
        "python": platform.python_version(),
    }


def write_results(path: str | Path, results: list[EngineResult]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([r.to_dict() for r in results], indent=2))
