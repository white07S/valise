#!/usr/bin/env python3
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Standalone microbenchmark for the valise Python bindings.

Measures the batching cliff (put_many vs per-call put), search/get throughput,
and prints a zero-copy evidence block. Exits 0 even when the rejection probes
fire (they are expected).

Run:
    python bench/bench.py [--n 10000] [--dim 64] [--path /tmp/bench.vls]

--dim must be a positive multiple of 64 (the engine rejects otherwise).
"""

import argparse
import os
import tempfile
import time

import numpy as np

import valise
from valise import Bm25, Rerank, Rrf


def matrix(n: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, dim)).astype(np.float32)
    m /= np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-6)
    return np.ascontiguousarray(m, dtype=np.float32)


class Table:
    """Plain, dependency-free fixed-width table."""

    COLS = ("op", "n", "total_ms", "per_op_us", "per_s")
    WIDTHS = (26, 10, 12, 12, 14)

    def __init__(self):
        self.rows = []

    def add(self, op, n, total_s):
        per_op_us = (total_s / n) * 1e6 if n else 0.0
        per_s = (n / total_s) if total_s else 0.0
        self.rows.append(
            (op, f"{n}", f"{total_s * 1e3:.2f}", f"{per_op_us:.2f}", f"{per_s:,.0f}")
        )

    def render(self):
        def fmt(cells):
            return "  ".join(str(c).ljust(w) for c, w in zip(cells, self.WIDTHS))

        lines = [fmt(self.COLS), fmt("-" * w for w in self.WIDTHS)]
        lines += [fmt(r) for r in self.rows]
        return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=10_000)
    ap.add_argument("--dim", type=int, default=64)
    ap.add_argument("--path", type=str, default=None)
    args = ap.parse_args()

    n, dim = args.n, args.dim
    if dim % 64 != 0 or dim <= 0:
        raise SystemExit(f"--dim must be a positive multiple of 64 (got {dim})")

    tmpdir = tempfile.mkdtemp(prefix="valise_bench_")
    base = args.path or os.path.join(tmpdir, "bench.vls")

    print(f"valise bench  n={n}  dim={dim}  format={valise.format_version()}")
    print()

    mat = matrix(n, dim, seed=0)
    keys = [f"k{i}" for i in range(n)]
    texts = ["quantum retrieval document"] * n
    table = Table()

    # ---- 1. Batch put_many + commit (staging vs commit barrier) ----
    s = valise.Store.create(base + ".batch")
    s.define_vector_space("emb", dim=dim, deferred_sample=min(n, 2000))
    s.define_text_space("en")
    s.collection("kb", valise.Shape().text("body", "en").vector("dense", "emb"))
    w = s.writer()
    t0 = time.perf_counter()
    w.put_many("kb", keys, mat, texts=texts)
    stage_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    w.commit()
    commit_s = time.perf_counter() - t0
    w.close()
    put_many_total = stage_s + commit_s
    table.add("put_many stage", n, stage_s)
    table.add("put_many commit barrier", n, commit_s)
    table.add("put_many total", n, put_many_total)

    # ---- 2. Per-call put baseline (fresh store) ----
    s2 = valise.Store.create(base + ".percall")
    s2.define_vector_space("emb", dim=dim, deferred_sample=min(n, 2000))
    s2.define_text_space("en")
    s2.collection("kb", valise.Shape().text("body", "en").vector("dense", "emb"))
    w2 = s2.writer()
    t0 = time.perf_counter()
    for i in range(n):
        row = np.ascontiguousarray(mat[i])
        w2.put("kb", keys[i], valise.Record().text("body", texts[i]).vector("dense", row))
    percall_stage_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    w2.commit()
    percall_commit_s = time.perf_counter() - t0
    w2.close()
    percall_total = percall_stage_s + percall_commit_s
    table.add("put per-call total", n, percall_total)

    speedup = percall_total / put_many_total if put_many_total else float("nan")

    # ---- 3. Vector search k=10 ----
    reader = s.reader()
    qn = 1000
    t0 = time.perf_counter()
    for i in range(qn):
        q = np.ascontiguousarray(mat[i % n])
        reader.search("kb", valise.Search().vector("dense", q, Rerank.FAST).top_k(10))
    search_s = time.perf_counter() - t0
    table.add("vector search k=10", qn, search_s)

    # ---- 3b. Hybrid RRF search ----
    t0 = time.perf_counter()
    for i in range(qn):
        q = np.ascontiguousarray(mat[i % n])
        reader.search(
            "kb",
            valise.Search()
            .text("body", "quantum", Bm25())
            .vector("dense", q, Rerank.FAST)
            .fuse(Rrf(60))
            .top_k(10),
        )
    hybrid_s = time.perf_counter() - t0
    table.add("hybrid RRF search k=10", qn, hybrid_s)

    # ---- 4. get ----
    gn = 1000
    t0 = time.perf_counter()
    for i in range(gn):
        reader.get("kb", keys[i % n])
    get_s = time.perf_counter() - t0
    table.add("get", gn, get_s)

    print(table.render())
    print()
    print(f"batching speedup (per-call / put_many): {speedup:.1f}x")
    print()

    # ---- 5. Zero-copy / boundary evidence (print pass/fail, never raise) ----
    print("zero-copy / boundary evidence")
    print("-" * 40)

    # contiguous float32 1-D accepted
    try:
        valise.Record().vector("dense", np.ascontiguousarray(mat[0]))
        print("  contiguous f32 1-D vector ......... OK")
    except Exception as e:  # pragma: no cover
        print(f"  contiguous f32 1-D vector ......... FAIL ({e!r})")

    # non-contiguous f32 into put_many -> rejected
    noncontig = mat[:, ::2] if dim >= 2 else mat
    try:
        w3 = s.writer()
        try:
            w3.put_many("kb", keys[: noncontig.shape[0]], noncontig)
            print("  non-contiguous put_many .......... UNEXPECTED accept")
        finally:
            w3.close()
    except valise.ValidationError:
        print("  non-contiguous put_many .......... rejected (expected)")
    except Exception as e:
        print(f"  non-contiguous put_many .......... rejected ({type(e).__name__}, expected)")

    # float64 into Record().vector -> rejected (TypeError or ValidationError)
    try:
        valise.Record().vector("dense", np.ones(dim, dtype=np.float64))
        print("  float64 vector ................... UNEXPECTED accept")
    except (TypeError, valise.ValidationError):
        print("  float64 vector ................... rejected (expected)")
    except Exception as e:  # pragma: no cover
        print(f"  float64 vector ................... rejected ({type(e).__name__}, expected)")

    print()
    print("done.")


if __name__ == "__main__":
    main()
