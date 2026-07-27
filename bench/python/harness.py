"""Timing / measurement helpers shared by every runner.

Latency is measured **one query at a time, warm cache** (matching how the
Rust/Valise bench measures it), so `p50/p95/p99` and single-thread `qps` are
directly comparable. Storage and peak RSS are sampled per the bench plan's
emphasis on footprint.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path

import numpy as np
import psutil


def percentiles(samples_us: list[float]) -> dict[str, float]:
    a = np.sort(np.asarray(samples_us, dtype=np.float64))
    n = len(a)
    if n == 0:
        return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0}

    def pick(p: float) -> float:
        return float(a[min(n - 1, int(round(p / 100.0 * (n - 1))))])

    return {"p50": pick(50), "p95": pick(95), "p99": pick(99), "mean": float(a.mean())}


def time_search(search_one, queries: np.ndarray, k: int, warmup: int = 64):
    """Warm up, then time each query individually.

    `search_one(vec, k) -> np.ndarray[int]` returns the retrieved corpus
    indices. Returns `(retrieved (nq, k) int32, samples_us, qps, res)` where
    `res` carries CPU pressure for the query phase: `cpu_seconds` and
    `effective_cores ≈ cpu_seconds / wall` (~1.0 = single-core).
    """
    nq = len(queries)
    for i in range(min(warmup, nq)):
        search_one(queries[i], k)

    proc = psutil.Process()
    c0 = proc.cpu_times()
    samples: list[float] = []
    retrieved = np.full((nq, k), -1, dtype=np.int32)
    for i in range(nq):
        t0 = time.perf_counter()
        ids = search_one(queries[i], k)
        samples.append((time.perf_counter() - t0) * 1e6)
        ids = np.asarray(ids, dtype=np.int32)[:k]
        retrieved[i, : len(ids)] = ids
    c1 = proc.cpu_times()

    total_s = sum(samples) / 1e6
    qps = nq / total_s if total_s > 0 else 0.0
    cpu_s = (c1.user + c1.system) - (c0.user + c0.system)
    res = {
        "cpu_seconds": cpu_s,
        "effective_cores": (cpu_s / total_s) if total_s > 0 else 0.0,
    }
    return retrieved, samples, qps, res


class ResourceSampler:
    """Context manager that samples process RSS in a background thread,
    reporting `peak` and `mean` bytes over the wrapped region. Used per
    engine (build + query) to get a per-engine memory-pressure number —
    `getrusage` maxrss can't do that when several engines share one process.
    """

    def __init__(self, interval: float = 0.05) -> None:
        self.proc = psutil.Process()
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.peak = 0
        self._sum = 0
        self._n = 0

    def __enter__(self) -> "ResourceSampler":
        rss = self.proc.memory_info().rss
        self.peak, self._sum, self._n = rss, rss, 1
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            rss = self.proc.memory_info().rss
            self.peak = max(self.peak, rss)
            self._sum += rss
            self._n += 1

    def __exit__(self, *exc) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)

    @property
    def mean(self) -> int:
        return int(self._sum / self._n) if self._n else self.peak


def prep_vectors(ds):
    """Writable, contiguous f32 corpus/queries; L2-normalized for cosine
    so an inner-product index reproduces cosine (matches the GT)."""
    xb = np.array(ds.corpus, dtype=np.float32, copy=True)
    xq = np.array(ds.queries, dtype=np.float32, copy=True)
    if ds.metric == "cosine":
        import faiss

        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
    return xb, xq


def dir_size(path: str | Path) -> int:
    """Total bytes of a file or directory tree (on-disk index size)."""
    p = Path(path)
    if p.is_file():
        return p.stat().st_size
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
