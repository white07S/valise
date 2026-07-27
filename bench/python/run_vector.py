# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Vector ANN runners: FAISS (Flat/IVF-PQ/OPQ-IVF-PQ/SQ8/RaBitQ/HNSW),
hnswlib, LanceDB, usearch (f32/i8/b1), sqlite-vec.

Each engine builds once, then sweeps its accuracy knob to trace a
recall/latency Pareto. Latency is per-query, single-thread, warm (FAISS
OMP pinned to 1 thread so a single query is single-core); Valise, by
contrast, parallelizes *within* a query — see the report notes. Storage
(on-disk index bytes + bytes/vector + a `bytes_include` disclosure of
what those bytes count), build time, and per-engine CPU (effective
cores) + peak RSS are recorded per the plan's footprint emphasis.

With `--isolate`, the driver re-executes itself once per engine so each
engine's peak RSS is measured in a fresh process (corpus load included
once per child — the paper's "peak RSS per engine in a fresh process"
semantics); the parent merges the children's rows into `--out`.

    uv run python run_vector.py --dataset cohere-medium-1m-f32 --n 100000 --nq 1000
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import valise_data as D
from harness import ResourceSampler, dir_size, percentiles, prep_vectors, time_search
from metrics import recall_at_k
from result import EngineResult, write_results


def _pick_pq_m(d: int) -> int:
    for m in (d // 8, d // 4, d // 16, 64, 32, 16, 8, 4, 2, 1):
        if m >= 1 and d % m == 0:
            return m
    return 1


def _base(ds, args, engine, knob_name, knob_value, lib_version) -> EngineResult:
    return EngineResult(
        engine=engine,
        modality="vector",
        dataset=ds.name,
        n=ds.n,
        nq=ds.nq,
        top_k=args.k,
        dim=ds.dim,
        metric=ds.metric,
        knob_name=knob_name,
        knob_value=knob_value,
        lib_version=lib_version,
        threads=1,
    )


def _fill(r, build_s, index_bytes, ret, gt, samples, qps, res) -> None:
    """Populate timing, storage, quality, and CPU fields on a result.
    (Peak RSS is set by the per-engine sampler in `main`.)"""
    r.build_seconds = build_s
    r.total_index_seconds = build_s
    r.index_bytes = index_bytes
    r.qps = qps
    r.cpu_seconds = res["cpu_seconds"]
    r.effective_cores = res["effective_cores"]
    pc = percentiles(samples)
    r.p50_us, r.p95_us, r.p99_us, r.mean_us = (
        pc["p50"],
        pc["p95"],
        pc["p99"],
        pc["mean"],
    )
    r.recall_at_1 = recall_at_k(ret, gt, 1)
    r.recall_at_10 = recall_at_k(ret, gt, 10)
    r.recall_at_100 = recall_at_k(ret, gt, 100)


# --------------------------------------------------------------------------
def run_faiss(ds, gt, xb, xq, args) -> list[EngineResult]:
    import faiss

    faiss.omp_set_num_threads(1)  # single-core per-query latency
    d = ds.dim
    ip = ds.metric == "cosine"
    metric = faiss.METRIC_INNER_PRODUCT if ip else faiss.METRIC_L2
    out: list[EngineResult] = []
    tmp = Path(tempfile.mkdtemp(prefix="faiss_"))

    # --- Flat: exact baseline (and a recall=1.0 latency anchor) ---
    flat = faiss.IndexFlatIP(d) if ip else faiss.IndexFlatL2(d)
    t0 = time.perf_counter()
    flat.add(xb)
    build_s = time.perf_counter() - t0
    faiss.write_index(flat, str(tmp / "flat.idx"))
    ret, samples, qps, res = time_search(
        lambda q, k: flat.search(q.reshape(1, -1), k)[1][0], xq, args.k, args.warmup
    )
    r = _base(ds, args, "faiss-flat", "exact", None, faiss.__version__)
    _fill(r, build_s, dir_size(tmp / "flat.idx"), ret, gt, samples, qps, res)
    r.bytes_include = "f32_vectors(serialized_index)"
    out.append(r)

    # --- IVF-PQ: sweep nprobe ---
    nlist = min(4096, max(64, int(4 * math.sqrt(ds.n))))
    m = _pick_pq_m(d)
    quant = faiss.IndexFlatIP(d) if ip else faiss.IndexFlatL2(d)
    ivf = faiss.IndexIVFPQ(quant, d, nlist, m, 8, metric)
    t0 = time.perf_counter()
    ivf.train(xb)
    ivf.add(xb)
    build_s = time.perf_counter() - t0
    faiss.write_index(ivf, str(tmp / "ivfpq.idx"))
    sz = dir_size(tmp / "ivfpq.idx")
    for nprobe in [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if p <= nlist]:
        ivf.nprobe = nprobe
        ret, samples, qps, res = time_search(
            lambda q, k: ivf.search(q.reshape(1, -1), k)[1][0], xq, args.k, args.warmup
        )
        r = _base(ds, args, "faiss-ivfpq", "nprobe", nprobe, faiss.__version__)
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = f"nlist={nlist} m={m} nbits=8"
        r.bytes_include = "pq_codes+ivf_lists+ids"
        out.append(r)

    # --- HNSW (flat): sweep efSearch ---
    hnsw = faiss.IndexHNSWFlat(d, 32, metric)
    hnsw.hnsw.efConstruction = 200
    t0 = time.perf_counter()
    hnsw.add(xb)
    build_s = time.perf_counter() - t0
    faiss.write_index(hnsw, str(tmp / "hnsw.idx"))
    sz = dir_size(tmp / "hnsw.idx")
    for ef in (16, 32, 64, 128, 256):
        hnsw.hnsw.efSearch = ef
        ret, samples, qps, res = time_search(
            lambda q, k: hnsw.search(q.reshape(1, -1), k)[1][0], xq, args.k, args.warmup
        )
        r = _base(ds, args, "faiss-hnsw", "efSearch", ef, faiss.__version__)
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = "M=32 efC=200 (stores full f32 vectors)"
        r.bytes_include = "graph+f32_vectors"
        out.append(r)
    return out


# --------------------------------------------------------------------------
def run_hnswlib(ds, gt, xb, xq, args) -> list[EngineResult]:
    import hnswlib

    space = "ip" if ds.metric == "cosine" else "l2"  # xb pre-normalized for cosine
    idx = hnswlib.Index(space=space, dim=ds.dim)
    idx.init_index(max_elements=ds.n, ef_construction=200, M=16)
    idx.set_num_threads(1)
    t0 = time.perf_counter()
    idx.add_items(xb, np.arange(ds.n))
    build_s = time.perf_counter() - t0
    tmp = Path(tempfile.mkdtemp(prefix="hnswlib_")) / "idx.bin"
    idx.save_index(str(tmp))
    sz = dir_size(tmp)

    out = []
    for ef in (16, 32, 64, 128, 256):
        idx.set_ef(ef)
        ret, samples, qps, res = time_search(
            lambda q, k: idx.knn_query(q.reshape(1, -1), k=k)[0][0],
            xq,
            args.k,
            args.warmup,
        )
        r = _base(ds, args, "hnswlib", "ef", ef, getattr(hnswlib, "__version__", "?"))
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = "M=16 efC=200 in-memory (full f32 in RAM)"
        r.bytes_include = "graph+f32_vectors+ids"
        out.append(r)
    return out


# --------------------------------------------------------------------------
def run_lancedb(ds, gt, xb, xq, args) -> list[EngineResult]:
    import lancedb
    import pyarrow as pa

    metric = "cosine" if ds.metric == "cosine" else "l2"
    dbdir = Path(tempfile.mkdtemp(prefix="lancedb_"))
    db = lancedb.connect(str(dbdir))
    vec = pa.FixedSizeListArray.from_arrays(pa.array(xb.reshape(-1)), ds.dim)
    table = pa.table({"idx": pa.array(np.arange(ds.n, dtype=np.int64)), "vector": vec})
    t0 = time.perf_counter()
    tbl = db.create_table("v", table)
    nlist = min(4096, max(64, int(4 * math.sqrt(ds.n))))
    m = _pick_pq_m(ds.dim)
    tbl.create_index(
        metric=metric, num_partitions=nlist, num_sub_vectors=m, index_type="IVF_PQ"
    )
    build_s = time.perf_counter() - t0
    sz = dir_size(dbdir)

    def make_search(nprobes):
        def search_one(q, k):
            rows = tbl.search(q).metric(metric).nprobes(nprobes).limit(k).to_list()
            return [row["idx"] for row in rows]

        return search_one

    out = []
    for nprobes in [p for p in (1, 5, 10, 20, 50, 100) if p <= nlist]:
        ret, samples, qps, res = time_search(
            make_search(nprobes), xq, args.k, args.warmup
        )
        r = _base(
            ds,
            args,
            "lancedb",
            "nprobes",
            nprobes,
            getattr(lancedb, "__version__", "?"),
        )
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = f"IVF_PQ nlist={nlist} m={m} (on-disk)"
        r.bytes_include = "lance_dataset_dir(f32_column+pq_codes+ivf+manifests)"
        out.append(r)
    return out


# --------------------------------------------------------------------------
def run_usearch(ds, gt, xb, xq, args) -> list[EngineResult]:
    import usearch
    from usearch.index import Index

    metric = "ip" if ds.metric == "cosine" else "l2sq"  # xb pre-normalized for cosine
    idx = Index(
        ndim=ds.dim, metric=metric, dtype="f32", connectivity=16, expansion_add=200
    )
    t0 = time.perf_counter()
    idx.add(np.arange(ds.n), xb, threads=1)
    build_s = time.perf_counter() - t0
    tmp = Path(tempfile.mkdtemp(prefix="usearch_")) / "idx.usearch"
    idx.save(str(tmp))
    sz = dir_size(tmp)

    out = []
    for ef in (16, 32, 64, 128, 256):
        idx.expansion_search = ef
        ret, samples, qps, res = time_search(
            lambda q, k: idx.search(q, k, threads=1).keys,
            xq,
            args.k,
            args.warmup,
        )
        r = _base(ds, args, "usearch", "ef", ef, getattr(usearch, "__version__", "?"))
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = "M=16 efC=200 dtype=f32 (memory-mapped)"
        r.bytes_include = "graph+f32_vectors+ids"
        out.append(r)
    return out


# --------------------------------------------------------------------------
def _nlist_for(n: int) -> int:
    """Same IVF nlist rule the plain IVF-PQ / LanceDB runners use."""
    return min(4096, max(64, int(4 * math.sqrt(n))))


def run_faiss_opq_ivfpq(ds, gt, xb, xq, args) -> list[EngineResult]:
    """OPQ rotation + IVF-PQ — same M / nlist / training data as the plain
    IVF-PQ runner, so any recall gain is attributable to the rotation."""
    import faiss

    faiss.omp_set_num_threads(1)
    d = ds.dim
    metric = faiss.METRIC_INNER_PRODUCT if ds.metric == "cosine" else faiss.METRIC_L2
    nlist = _nlist_for(ds.n)
    m = _pick_pq_m(d)
    factory = f"OPQ{m},IVF{nlist},PQ{m}x8"
    index = faiss.index_factory(d, factory, metric)
    t0 = time.perf_counter()
    index.train(xb)  # same corpus the plain IVF-PQ trains on
    index.add(xb)
    build_s = time.perf_counter() - t0
    tmp = Path(tempfile.mkdtemp(prefix="faiss_opq_"))
    faiss.write_index(index, str(tmp / "opq_ivfpq.idx"))
    sz = dir_size(tmp / "opq_ivfpq.idx")
    ivf = faiss.extract_index_ivf(index)

    out = []
    for nprobe in [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if p <= nlist]:
        ivf.nprobe = nprobe
        ret, samples, qps, res = time_search(
            lambda q, k: index.search(q.reshape(1, -1), k)[1][0],
            xq,
            args.k,
            args.warmup,
        )
        r = _base(ds, args, "faiss-opq-ivfpq", "nprobe", nprobe, faiss.__version__)
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = f"index_factory('{factory}') nlist={nlist} m={m} nbits=8"
        r.bytes_include = "opq_matrix+pq_codes+ivf_lists+ids"
        out.append(r)
    return out


def run_faiss_sq8(ds, gt, xb, xq, args) -> list[EngineResult]:
    """IVF + 8-bit scalar quantizer (sweep nprobe) plus a flat SQ8 point."""
    import faiss

    faiss.omp_set_num_threads(1)
    d = ds.dim
    metric = faiss.METRIC_INNER_PRODUCT if ds.metric == "cosine" else faiss.METRIC_L2
    nlist = _nlist_for(ds.n)
    tmp = Path(tempfile.mkdtemp(prefix="faiss_sq8_"))
    out = []

    # --- IVF,SQ8: sweep nprobe ---
    factory = f"IVF{nlist},SQ8"
    ivf = faiss.index_factory(d, factory, metric)
    t0 = time.perf_counter()
    ivf.train(xb)
    ivf.add(xb)
    build_s = time.perf_counter() - t0
    faiss.write_index(ivf, str(tmp / "ivfsq8.idx"))
    sz = dir_size(tmp / "ivfsq8.idx")
    for nprobe in [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if p <= nlist]:
        ivf.nprobe = nprobe
        ret, samples, qps, res = time_search(
            lambda q, k: ivf.search(q.reshape(1, -1), k)[1][0], xq, args.k, args.warmup
        )
        r = _base(ds, args, "faiss-sq8", "nprobe", nprobe, faiss.__version__)
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = f"index_factory('{factory}')"
        r.bytes_include = "sq8_codes+ivf_lists+ids"
        out.append(r)

    # --- flat SQ8: exhaustive scan over 8-bit codes, single point ---
    flat = faiss.index_factory(d, "SQ8", metric)
    t0 = time.perf_counter()
    flat.train(xb)
    flat.add(xb)
    build_s = time.perf_counter() - t0
    faiss.write_index(flat, str(tmp / "sq8flat.idx"))
    szf = dir_size(tmp / "sq8flat.idx")
    ret, samples, qps, res = time_search(
        lambda q, k: flat.search(q.reshape(1, -1), k)[1][0], xq, args.k, args.warmup
    )
    r = _base(ds, args, "faiss-sq8-flat", "exact-scan", None, faiss.__version__)
    _fill(r, build_s, szf, ret, gt, samples, qps, res)
    r.notes = "index_factory('SQ8') brute-force over 8-bit codes"
    r.bytes_include = "sq8_codes(serialized_index)"
    out.append(r)
    return out


def run_faiss_rabitq(ds, gt, xb, xq, args) -> list[EngineResult]:
    """FAISS 1.14 RaBitQ (1-bit codes + per-vector factors), IVF variant.

    Sweeps nprobe at the default query-quantization qb, then sweeps qb at
    a fixed mid nprobe (qb is a pure query-time knob on IndexIVFRaBitQ;
    qb=0 means unquantized f32 queries).
    """
    import faiss

    faiss.omp_set_num_threads(1)
    d = ds.dim
    metric = faiss.METRIC_INNER_PRODUCT if ds.metric == "cosine" else faiss.METRIC_L2
    nlist = _nlist_for(ds.n)
    factory = f"IVF{nlist},RaBitQ"
    index = faiss.index_factory(d, factory, metric)  # faiss.IndexIVFRaBitQ
    default_qb = int(index.qb)
    t0 = time.perf_counter()
    index.train(xb)
    index.add(xb)
    build_s = time.perf_counter() - t0
    tmp = Path(tempfile.mkdtemp(prefix="faiss_rabitq_"))
    faiss.write_index(index, str(tmp / "ivfrabitq.idx"))
    sz = dir_size(tmp / "ivfrabitq.idx")

    out = []
    probes = [p for p in (1, 2, 4, 8, 16, 32, 64, 128) if p <= nlist]
    for nprobe in probes:
        index.nprobe = nprobe
        ret, samples, qps, res = time_search(
            lambda q, k: index.search(q.reshape(1, -1), k)[1][0],
            xq,
            args.k,
            args.warmup,
        )
        r = _base(ds, args, "faiss-rabitq", "nprobe", nprobe, faiss.__version__)
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = f"index_factory('{factory}') qb={default_qb}"
        r.bytes_include = "rabitq_1bit_codes+factors+ivf_lists+ids"
        out.append(r)

    # qb sweep at a fixed nprobe (query-side only; index unchanged).
    qb_nprobe = 32 if 32 in probes else probes[-1]
    index.nprobe = qb_nprobe
    for qb in (0, 2, 4, 8):
        if qb == default_qb:
            continue  # already covered by the nprobe sweep at this nprobe
        index.qb = qb
        ret, samples, qps, res = time_search(
            lambda q, k: index.search(q.reshape(1, -1), k)[1][0],
            xq,
            args.k,
            args.warmup,
        )
        r = _base(ds, args, "faiss-rabitq", "qb", qb, faiss.__version__)
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = f"index_factory('{factory}') nprobe={qb_nprobe} (qb=0: f32 queries)"
        r.bytes_include = "rabitq_1bit_codes+factors+ivf_lists+ids"
        out.append(r)
    index.qb = default_qb
    return out


# --------------------------------------------------------------------------
def _run_usearch_quant(ds, gt, xb, xq, args, engine: str, kind: str):
    """usearch HNSW with quantized storage (i8 or b1); same M/efC and ef
    sweep as the f32 usearch runner. Input transforms are order-preserving
    for the dataset's metric and disclosed in `notes`."""
    import usearch
    from usearch.index import Index, ScalarKind

    if kind == "i8":
        # 'cos' (not 'ip') for cosine data: usearch's i8 distance under
        # 'cos' corrects for the quantized norms and measurably beats raw
        # integer IP (R@10 0.88 vs 0.81 on cohere-10k). For L2 data,
        # rescale by corpus max-abs so values fit the cast's assumed
        # [-1, 1] range (a uniform scale preserves L2 order).
        if ds.metric == "cosine":
            metric = "cos"
            vb, vq = xb, xq
            note = "M=16 efC=200 dtype=i8 metric=cos"
        else:
            metric = "l2sq"
            s = float(np.abs(xb).max())
            vb, vq = xb / s, xq / s
            note = f"M=16 efC=200 dtype=i8 l2sq (inputs/{s:.3g}: order-preserving)"
        idx = Index(
            ndim=ds.dim,
            metric=metric,
            dtype=ScalarKind.I8,
            connectivity=16,
            expansion_add=200,
        )
        binc = "graph+i8_vectors+ids"
    elif kind == "b1":
        # 1 bit/dim sign sketch, searched under hamming. For cosine data
        # sign(x) is the classic angular sketch; for L2 data we center per
        # dimension first (translation preserves L2 order) so signs carry
        # information (raw SIFT components are all >= 0).
        if ds.metric == "cosine":
            vb = np.packbits(xb > 0, axis=1)
            vq = np.packbits(xq > 0, axis=1)
            note = "M=16 efC=200 dtype=b1 hamming (sign-bit angular sketch)"
        else:
            mu = xb.mean(axis=0)
            vb = np.packbits((xb - mu) > 0, axis=1)
            vq = np.packbits((xq - mu) > 0, axis=1)
            note = "M=16 efC=200 dtype=b1 hamming (mean-centered sign bits)"
        idx = Index(
            ndim=ds.dim,
            metric="hamming",
            dtype=ScalarKind.B1,
            connectivity=16,
            expansion_add=200,
        )
        binc = "graph+b1_packed_bits+ids"
    else:  # pragma: no cover
        raise ValueError(kind)

    t0 = time.perf_counter()
    idx.add(np.arange(ds.n), vb, threads=1)
    build_s = time.perf_counter() - t0
    tmp = Path(tempfile.mkdtemp(prefix=f"usearch_{kind}_")) / "idx.usearch"
    idx.save(str(tmp))
    sz = dir_size(tmp)

    out = []
    for ef in (16, 32, 64, 128, 256):
        idx.expansion_search = ef
        ret, samples, qps, res = time_search(
            lambda q, k: idx.search(q, k, threads=1).keys,
            vq,
            args.k,
            args.warmup,
        )
        r = _base(ds, args, engine, "ef", ef, getattr(usearch, "__version__", "?"))
        _fill(r, build_s, sz, ret, gt, samples, qps, res)
        r.notes = note
        r.bytes_include = binc
        out.append(r)
    return out


def run_usearch_i8(ds, gt, xb, xq, args) -> list[EngineResult]:
    return _run_usearch_quant(ds, gt, xb, xq, args, "usearch-i8", "i8")


def run_usearch_b1(ds, gt, xb, xq, args) -> list[EngineResult]:
    return _run_usearch_quant(ds, gt, xb, xq, args, "usearch-b1", "b1")


# --------------------------------------------------------------------------
def run_sqlite_vec(ds, gt, xb, xq, args) -> list[EngineResult]:
    """sqlite-vec vec0: embedded single-file exact brute-force peer.

    Vectors are pre-normalized for cosine (harness), so vec0's default L2
    ranking reproduces cosine order — recall should be ~1.0. One operating
    point, plus a 1-bit `bit[d]` / vec_quantize_binary hamming variant.
    Index bytes = whole .db file (sqlite pages), the honest single-file
    accounting.
    """
    import sqlite3

    import sqlite_vec

    d = ds.dim
    tmpdir = Path(tempfile.mkdtemp(prefix="sqlitevec_"))
    ver = getattr(sqlite_vec, "__version__", "?")
    out = []

    def open_db(path: Path) -> sqlite3.Connection:
        db = sqlite3.connect(str(path))
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        return db

    # --- float32 exact ---
    fpath = tmpdir / "vec_f32.db"
    db = open_db(fpath)
    t0 = time.perf_counter()
    db.execute(f"CREATE VIRTUAL TABLE v USING vec0(embedding float[{d}])")
    with db:  # single transaction: this *is* the build
        db.executemany(
            "INSERT INTO v(rowid, embedding) VALUES (?, ?)",
            ((i, xb[i].tobytes()) for i in range(ds.n)),
        )
    build_s = time.perf_counter() - t0
    sz = dir_size(fpath)
    sql = "SELECT rowid FROM v WHERE embedding MATCH ? AND k = ? ORDER BY distance"
    ret, samples, qps, res = time_search(
        lambda q, k: [row[0] for row in db.execute(sql, (q.tobytes(), k))],
        xq,
        args.k,
        args.warmup,
    )
    r = _base(ds, args, "sqlite-vec", "exact", None, ver)
    _fill(r, build_s, sz, ret, gt, samples, qps, res)
    r.notes = f"vec0 float[{d}] brute force" + (
        "; L2 on pre-normalized vectors == cosine" if ds.metric == "cosine" else ""
    )
    r.bytes_include = "sqlite_pages_whole_file"
    out.append(r)
    db.close()

    # --- 1-bit quantized variant (vec_quantize_binary, hamming) ---
    # vec_quantize_binary takes sign(x): fine for cosine data; for L2 data
    # center per dimension first (translation preserves L2 order — raw
    # SIFT components are all >= 0, so uncentered signs carry ~no signal).
    if ds.metric == "cosine":
        bb, bq = xb, xq
        centered = ""
    else:
        mu = xb.mean(axis=0)
        bb, bq = xb - mu, xq - mu
        centered = ", mean-centered inputs"
    bpath = tmpdir / "vec_bit.db"
    db = open_db(bpath)
    t0 = time.perf_counter()
    db.execute(f"CREATE VIRTUAL TABLE vb USING vec0(embedding bit[{d}])")
    with db:
        db.executemany(
            "INSERT INTO vb(rowid, embedding) VALUES (?, vec_quantize_binary(?))",
            ((i, bb[i].tobytes()) for i in range(ds.n)),
        )
    build_s = time.perf_counter() - t0
    szb = dir_size(bpath)
    sqlb = (
        "SELECT rowid FROM vb WHERE embedding MATCH vec_quantize_binary(?) "
        "AND k = ? ORDER BY distance"
    )
    ret, samples, qps, res = time_search(
        lambda q, k: [row[0] for row in db.execute(sqlb, (q.tobytes(), k))],
        bq,
        args.k,
        args.warmup,
    )
    r = _base(ds, args, "sqlite-vec-bit", "exact-scan", None, ver)
    _fill(r, build_s, szb, ret, gt, samples, qps, res)
    r.notes = f"vec0 bit[{d}] vec_quantize_binary, hamming brute force{centered}"
    r.bytes_include = "sqlite_pages_whole_file"
    out.append(r)
    db.close()
    return out


ENGINES = {
    "faiss": run_faiss,
    "faiss-opq-ivfpq": run_faiss_opq_ivfpq,
    "faiss-sq8": run_faiss_sq8,
    "faiss-rabitq": run_faiss_rabitq,
    "hnswlib": run_hnswlib,
    "lancedb": run_lancedb,
    "usearch": run_usearch,
    "usearch-i8": run_usearch_i8,
    "usearch-b1": run_usearch_b1,
    "sqlite-vec": run_sqlite_vec,
}


def _run_isolated(args) -> None:
    """Fresh process per engine: re-exec this script with a single-engine
    `--engines` and an internal `--child-out`, then merge the children's
    rows into `--out`. Each child loads the corpus itself, so its peak RSS
    is an honest per-engine, fresh-process number."""
    # Warm the ground-truth cache once so no single child pays (or times)
    # the exact-GT computation; children then just np.load it.
    ds = D.load_vectors(args.dataset, n=args.n, nq=args.nq)
    t0 = time.perf_counter()
    gt = D.ground_truth(ds, k=args.k)
    print(f"[gt] warmed cache {gt.shape} in {time.perf_counter() - t0:.1f}s")
    del gt, ds

    tmpdir = Path(tempfile.mkdtemp(prefix="valise_isolate_"))
    all_rows: list[dict] = []
    for i, name in enumerate(n.strip() for n in args.engines.split(",")):
        if name not in ENGINES:
            print(f"[skip] unknown engine {name!r}")
            continue
        child_out = tmpdir / f"{i}_{name}.json"
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--dataset", args.dataset,
            "--n", str(args.n),
            "--nq", str(args.nq),
            "--k", str(args.k),
            "--warmup", str(args.warmup),
            "--engines", name,
            "--child-out", str(child_out),
        ]
        print(f"[isolate] {name} -> fresh process")
        rc = subprocess.run(cmd).returncode
        if rc != 0 or not child_out.exists():
            print(f"[FAIL] {name}: child exited rc={rc}")
            continue
        all_rows.extend(json.loads(child_out.read_text()))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_rows, indent=2))
    _print_table(all_rows)
    print(f"\n[wrote] {args.out} ({len(all_rows)} rows, isolated per engine)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="cohere-medium-1m-f32")
    ap.add_argument("--n", type=int, default=100_000)
    ap.add_argument("--nq", type=int, default=1000)
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=64)
    ap.add_argument("--engines", default="faiss,hnswlib,lancedb")
    ap.add_argument("--out", default="results/vector.json")
    ap.add_argument(
        "--isolate",
        action="store_true",
        help="run each engine in a fresh subprocess (honest per-engine peak RSS)",
    )
    ap.add_argument("--child-out", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.isolate and not args.child_out:
        _run_isolated(args)
        return

    ds = D.load_vectors(args.dataset, n=args.n, nq=args.nq)
    print(f"[load] {ds.name} n={ds.n} dim={ds.dim} nq={ds.nq} metric={ds.metric}")
    t0 = time.perf_counter()
    gt = D.ground_truth(ds, k=args.k)
    print(f"[gt] {gt.shape} in {time.perf_counter() - t0:.1f}s")
    xb, xq = prep_vectors(ds)

    results: list[EngineResult] = []
    for name in args.engines.split(","):
        name = name.strip()
        if name not in ENGINES:
            print(f"[skip] unknown engine {name!r}")
            continue
        print(f"[run] {name} ...")
        try:
            t0 = time.perf_counter()
            # Sample RSS over the whole engine (build + sweep) for a
            # per-engine memory-pressure number.
            with ResourceSampler() as rss:
                rows = ENGINES[name](ds, gt, xb, xq, args)
            for r in rows:
                r.peak_rss_bytes = rss.peak
                r.rss_mean_bytes = rss.mean
            results.extend(rows)
            print(
                f"[done] {name}: {len(rows)} rows in {time.perf_counter() - t0:.1f}s "
                f"(peak RSS {rss.peak / 1024 / 1024:.0f} MiB)"
            )
        except Exception as e:  # isolate per-engine failures
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    out_path = args.child_out or args.out
    write_results(out_path, results)
    _print_table([r.to_dict() for r in results])
    print(f"\n[wrote] {out_path}")


def _print_table(rows: list[dict]) -> None:
    print(
        f"\n{'engine':<16}{'knob':>10}{'R@10':>8}{'R@100':>8}{'p50us':>9}{'p95us':>9}"
        f"{'qps':>8}{'B/vec':>8}{'build_s':>9}{'cores':>7}{'RSS_MiB':>9}"
    )
    for r in rows:
        knob = (
            "exact"
            if r.get("knob_value") is None
            else f"{r['knob_name'][:4]}={int(r['knob_value'])}"
        )
        rss = (r.get("peak_rss_bytes") or 0) / 1024 / 1024
        print(
            f"{r['engine']:<16}{knob:>10}"
            f"{(r.get('recall_at_10') or 0):>8.3f}{(r.get('recall_at_100') or 0):>8.3f}"
            f"{(r.get('p50_us') or 0):>9.0f}{(r.get('p95_us') or 0):>9.0f}"
            f"{(r.get('qps') or 0):>8.0f}{(r.get('bytes_per_item') or 0):>8.0f}"
            f"{(r.get('build_seconds') or 0):>9.1f}"
            f"{(r.get('effective_cores') or 0):>7.2f}{rss:>9.0f}"
        )


if __name__ == "__main__":
    main()
