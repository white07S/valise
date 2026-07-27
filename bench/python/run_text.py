# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Text BM25 runners: bm25s (fast scipy-sparse), rank_bm25 (pure-python floor).

Pyserini (Lucene) is supported if installed (`--extra pyserini`, needs Java 21)
but optional — it's heavier to provision. Latency is per-query (tokenize +
retrieve), warm. Quality is nDCG@10 / MAP / MRR / Recall@100 vs BEIR qrels.

    uv run python run_text.py --dataset scifact --engines bm25s,rank_bm25
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time
from pathlib import Path

os.environ.setdefault("TQDM_DISABLE", "1")  # silence bm25s per-query progress bars

import numpy as np
import psutil

import valise_data as D
from harness import ResourceSampler, dir_size, percentiles
from metrics import text_metrics
from result import EngineResult, write_results


def _eval_and_pack(
    engine,
    bd,
    args,
    build_s,
    persist_s,
    index_bytes,
    samples,
    run,
    lib_version,
    cpu_seconds=None,
):
    r = EngineResult(
        engine=engine,
        modality="text",
        dataset=bd.name,
        n=len(bd.corpus),
        nq=len(bd.queries),
        top_k=args.k,
        lib_version=lib_version,
        threads=1,
    )
    r.build_seconds = build_s
    r.persist_seconds = persist_s
    r.total_index_seconds = (build_s or 0) + (persist_s or 0)
    r.index_bytes = index_bytes
    # peak_rss_bytes is set by the per-engine sampler in main().
    pc = percentiles(samples)
    r.p50_us, r.p95_us, r.p99_us, r.mean_us = (
        pc["p50"],
        pc["p95"],
        pc["p99"],
        pc["mean"],
    )
    total_s = sum(samples) / 1e6
    r.qps = len(samples) / total_s if total_s else 0.0
    r.cpu_seconds = cpu_seconds
    r.effective_cores = (cpu_seconds / total_s) if (cpu_seconds and total_s) else None
    m = text_metrics(run, bd.qrels)
    r.ndcg_at_10, r.map_score, r.mrr_at_10, r.recall_at_100_text = (
        m["ndcg_at_10"],
        m["map"],
        m["mrr"],
        m["recall_at_100"],
    )
    return r


def run_bm25s(bd, args) -> EngineResult:
    import bm25s

    ids = list(bd.corpus)
    texts = [bd.corpus[i] for i in ids]
    ctok = bm25s.tokenize(texts, stopwords="en", show_progress=False)
    bm = bm25s.BM25()
    t0 = time.perf_counter()
    bm.index(ctok, show_progress=False)
    build_s = time.perf_counter() - t0

    tmp = Path(tempfile.mkdtemp(prefix="bm25s_"))
    t0 = time.perf_counter()
    bm.save(str(tmp))
    persist_s = time.perf_counter() - t0
    index_bytes = dir_size(tmp)

    qids = list(bd.queries)
    # warmup
    for qid in qids[: min(args.warmup, len(qids))]:
        bm.retrieve(
            bm25s.tokenize([bd.queries[qid]], stopwords="en", show_progress=False),
            k=args.k,
        )

    samples: list[float] = []
    run: dict[str, dict[str, float]] = {}
    proc = psutil.Process()
    c0 = proc.cpu_times()
    for qid in qids:
        t0 = time.perf_counter()
        qtok = bm25s.tokenize([bd.queries[qid]], stopwords="en", show_progress=False)
        res, sc = bm.retrieve(qtok, k=min(args.k, len(ids)), show_progress=False)
        samples.append((time.perf_counter() - t0) * 1e6)
        run[qid] = {ids[int(res[0][j])]: float(sc[0][j]) for j in range(res.shape[1])}
    c1 = proc.cpu_times()
    cpu_s = (c1.user + c1.system) - (c0.user + c0.system)
    return _eval_and_pack(
        "bm25s",
        bd,
        args,
        build_s,
        persist_s,
        index_bytes,
        samples,
        run,
        getattr(bm25s, "__version__", "?"),
        cpu_seconds=cpu_s,
    )


def run_rank_bm25(bd, args) -> EngineResult:
    from rank_bm25 import BM25Okapi

    ids = list(bd.corpus)
    toks = [bd.corpus[i].lower().split() for i in ids]
    t0 = time.perf_counter()
    bm = BM25Okapi(toks)
    build_s = time.perf_counter() - t0  # in-memory only; no persist

    qids = list(bd.queries)
    for qid in qids[: min(args.warmup, len(qids))]:
        bm.get_scores(bd.queries[qid].lower().split())

    k = min(args.k, len(ids))
    samples: list[float] = []
    run: dict[str, dict[str, float]] = {}
    proc = psutil.Process()
    c0 = proc.cpu_times()
    for qid in qids:
        t0 = time.perf_counter()
        sc = bm.get_scores(bd.queries[qid].lower().split())
        top = np.argpartition(-sc, k - 1)[:k]
        top = top[np.argsort(-sc[top])]
        samples.append((time.perf_counter() - t0) * 1e6)
        run[qid] = {ids[int(j)]: float(sc[j]) for j in top}
    c1 = proc.cpu_times()
    cpu_s = (c1.user + c1.system) - (c0.user + c0.system)
    r = _eval_and_pack(
        "rank_bm25",
        bd,
        args,
        build_s,
        None,
        None,
        samples,
        run,
        "0.2",
        cpu_seconds=cpu_s,
    )
    r.notes = "pure-python, in-memory (no on-disk index)"
    return r


def run_pyserini(bd, args) -> EngineResult:
    import json as _json
    import subprocess
    import sys

    # pyserini eagerly builds an OpenAI client at import and needs a JDK
    # jnius can load — set both before importing its lucene modules.
    os.environ.setdefault("OPENAI_API_KEY", "sk-dummy")
    if "JAVA_HOME" not in os.environ:
        try:
            prefix = subprocess.check_output(
                ["brew", "--prefix", "openjdk@21"], text=True
            ).strip()
            jh = Path(prefix) / "libexec" / "openjdk.jdk" / "Contents" / "Home"
            if jh.exists():
                os.environ["JAVA_HOME"] = str(jh)
        except Exception:
            pass

    work = Path(tempfile.mkdtemp(prefix="pyserini_"))
    input_dir = work / "input"
    input_dir.mkdir()
    index_dir = work / "index"
    ids = list(bd.corpus)
    with (input_dir / "docs.jsonl").open("w") as f:
        for did in ids:
            f.write(_json.dumps({"id": did, "contents": bd.corpus[did]}) + "\n")

    # Build the Lucene index (JVM, multi-threaded — a one-time cost).
    t0 = time.perf_counter()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyserini.index.lucene",
            "--collection",
            "JsonCollection",
            "--input",
            str(input_dir),
            "--index",
            str(index_dir),
            "--generator",
            "DefaultLuceneDocumentGenerator",
            "--threads",
            "4",
        ],
        check=True,
        capture_output=True,
        env=os.environ,
    )
    build_s = time.perf_counter() - t0
    index_bytes = dir_size(index_dir)

    from pyserini.search.lucene import LuceneSearcher

    searcher = LuceneSearcher(str(index_dir))
    searcher.set_bm25(k1=0.9, b=0.4)  # Anserini/BEIR default

    def search_one(qtext):
        try:
            return searcher.search(qtext, k=args.k)
        except Exception:
            return []

    qids = list(bd.queries)
    for qid in qids[: min(args.warmup, len(qids))]:
        search_one(bd.queries[qid])

    samples: list[float] = []
    run: dict[str, dict[str, float]] = {}
    proc = psutil.Process()
    c0 = proc.cpu_times()
    for qid in qids:
        t0 = time.perf_counter()
        hits = search_one(bd.queries[qid])
        samples.append((time.perf_counter() - t0) * 1e6)
        run[qid] = {h.docid: float(h.score) for h in hits}
    c1 = proc.cpu_times()
    cpu_s = (c1.user + c1.system) - (c0.user + c0.system)

    try:
        import importlib.metadata as _md

        ver = _md.version("pyserini")
    except Exception:
        ver = "?"
    r = _eval_and_pack(
        "pyserini",
        bd,
        args,
        build_s,
        None,
        index_bytes,
        samples,
        run,
        ver,
        cpu_seconds=cpu_s,
    )
    r.notes = "Lucene BM25 k1=0.9 b=0.4 (JVM, openjdk@21); index threads=4"
    return r


ENGINES = {"bm25s": run_bm25s, "rank_bm25": run_rank_bm25, "pyserini": run_pyserini}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="scifact")
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=32)
    ap.add_argument("--engines", default="bm25s,rank_bm25")
    ap.add_argument("--out", default="results/text.json")
    args = ap.parse_args()

    bd = D.load_beir(args.dataset)
    print(f"[load] {bd.name} docs={len(bd.corpus)} judged_queries={len(bd.queries)}")

    results = []
    for name in args.engines.split(","):
        name = name.strip()
        if name not in ENGINES:
            print(f"[skip] unknown engine {name!r}")
            continue
        print(f"[run] {name} ...")
        try:
            t0 = time.perf_counter()
            with ResourceSampler() as rss:
                r = ENGINES[name](bd, args)
            r.peak_rss_bytes = rss.peak
            r.rss_mean_bytes = rss.mean
            results.append(r)
            print(
                f"[done] {name} in {time.perf_counter() - t0:.1f}s (peak RSS {rss.peak / 1024 / 1024:.0f} MiB)"
            )
        except Exception as e:
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    write_results(args.out, results)
    print(
        f"\n{'engine':<12}{'nDCG@10':>9}{'MAP':>8}{'MRR':>8}{'R@100':>8}{'p50us':>9}{'p95us':>9}"
        f"{'qps':>8}{'B/doc':>8}{'build_s':>9}{'cores':>7}{'RSS_MiB':>9}"
    )
    for r in results:
        rssm = (r.peak_rss_bytes or 0) / 1024 / 1024
        print(
            f"{r.engine:<12}{(r.ndcg_at_10 or 0):>9.4f}{(r.map_score or 0):>8.4f}{(r.mrr_at_10 or 0):>8.4f}"
            f"{(r.recall_at_100_text or 0):>8.4f}{(r.p50_us or 0):>9.0f}{(r.p95_us or 0):>9.0f}"
            f"{(r.qps or 0):>8.0f}{(r.bpv or 0):>8.0f}{(r.build_seconds or 0):>9.2f}"
            f"{(r.effective_cores or 0):>7.2f}{rssm:>9.0f}"
        )
    print(f"\n[wrote] {args.out}")


if __name__ == "__main__":
    main()
