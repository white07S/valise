"""Dataset loaders + shared ground truth for the Valise Python bench.

Vector data lives in `bench/datasets/<name>/` as row-major little-endian
f32 (`corpus.f32`, `queries.f32`) with a `meta.json`. SIFT/GIST also ship
an official `gt.u32` (true top-`gt_k` neighbors over the *full* corpus).

Ground truth (`ground_truth`) is the single source every engine — Valise
included — is graded against, so recall numbers are comparable:

* full corpus + official `gt.u32` present  -> use it (no compute).
* otherwise                                -> compute exact top-k with
  FAISS-Flat (IP on L2-normalized vectors for cosine; L2 otherwise) and
  cache to `cache/` keyed by (name, n, nq, metric, k).

Text data lives in `bench/beir-data/<name>/` (BEIR layout: `corpus.jsonl`,
`queries.jsonl`, `qrels/<split>.tsv`); qrels are the ground truth, so no
computation is needed.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
BENCH_DIR = _HERE.parent  # .../bench
VEC_DIR = BENCH_DIR / "datasets"
BEIR_DIR = BENCH_DIR / "beir-data"
CACHE = _HERE / "cache"


def metric_for(name: str) -> str:
    """Distance metric a vector dataset is evaluated under.

    Cohere/OpenAI/MPNet embeddings are compared by cosine; GloVe is the
    ann-benchmarks "angular" split (= cosine); the texmex SIFT/GIST
    benchmarks are L2. (Valise's Cohere space uses `VectorMetric::Cosine`.)
    """
    if name.startswith(("cohere", "openai", "glove", "mpnet")):
        return "cosine"
    if name.startswith(("sift", "gist")):
        return "l2"
    raise ValueError(f"unknown metric for dataset {name!r}; add it to metric_for()")


# --------------------------------------------------------------------------
# Vector
# --------------------------------------------------------------------------
@dataclass
class VectorData:
    name: str
    dim: int
    metric: str  # "cosine" | "l2"
    corpus: np.ndarray  # (n, dim) f32, read-only memmap slice
    queries: np.ndarray  # (nq, dim) f32, read-only memmap slice
    corpus_len: int  # full corpus length (official-gt validity check)
    query_len: int  # full query count
    gt_k: int | None = None  # cols in official gt.u32, if any

    @property
    def n(self) -> int:
        return int(self.corpus.shape[0])

    @property
    def nq(self) -> int:
        return int(self.queries.shape[0])


def available_vector_datasets() -> list[str]:
    return sorted(p.name for p in VEC_DIR.iterdir() if (p / "meta.json").exists())


def load_vectors(name: str, n: int | None = None, nq: int | None = None) -> VectorData:
    d = VEC_DIR / name
    meta = json.loads((d / "meta.json").read_text())
    dim, clen, qlen = meta["dim"], meta["corpus_len"], meta["query_len"]
    corpus = np.memmap(d / "corpus.f32", dtype="<f4", mode="r", shape=(clen, dim))
    queries = np.memmap(d / "queries.f32", dtype="<f4", mode="r", shape=(qlen, dim))
    n = clen if n is None else min(n, clen)
    nq = qlen if nq is None else min(nq, qlen)
    return VectorData(
        name=name,
        dim=dim,
        metric=metric_for(name),
        corpus=corpus[:n],
        queries=queries[:nq],
        corpus_len=clen,
        query_len=qlen,
        gt_k=meta.get("gt_k"),
    )


def ground_truth(ds: VectorData, k: int = 100) -> np.ndarray:
    """Exact top-`k` neighbor indices, shape (nq, k), int32. Cached."""
    official = VEC_DIR / ds.name / "gt.u32"
    if ds.n == ds.corpus_len and official.exists():
        gt_k = ds.gt_k or (official.stat().st_size // 4 // ds.query_len)
        gt = np.fromfile(official, dtype="<u4").reshape(ds.query_len, gt_k)
        if k > gt_k:
            raise ValueError(f"requested k={k} > official gt_k={gt_k} for {ds.name}")
        return gt[: ds.nq, :k].astype(np.int32)

    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"gt_{ds.name}_n{ds.n}_nq{ds.nq}_{ds.metric}_k{k}.npy"
    if cache.exists():
        return np.load(cache)
    gt = _compute_gt_faiss(ds, k)
    np.save(cache, gt)
    return gt


def _compute_gt_faiss(ds: VectorData, k: int) -> np.ndarray:
    import faiss

    # `np.array(..., copy=True)`, NOT `ascontiguousarray`: the corpus/query
    # arrays are read-only memmap slices, and `ascontiguousarray` returns
    # them unchanged when already contiguous. `faiss.normalize_L2` writes
    # in place, which would SIGBUS on a read-only mmap — so we own a
    # writable copy here.
    xb = np.array(ds.corpus, dtype=np.float32, copy=True)
    xq = np.array(ds.queries, dtype=np.float32, copy=True)
    if ds.metric == "cosine":
        faiss.normalize_L2(xb)
        faiss.normalize_L2(xq)
        index = faiss.IndexFlatIP(ds.dim)
    else:
        index = faiss.IndexFlatL2(ds.dim)
    index.add(xb)
    _, idx = index.search(xq, k)
    return idx.astype(np.int32)


# --------------------------------------------------------------------------
# Text (BEIR)
# --------------------------------------------------------------------------
@dataclass
class BeirData:
    name: str
    corpus: dict[str, str]  # doc_id -> "title. text"
    queries: dict[str, str]  # qid -> text (only queries that have qrels)
    qrels: dict[str, dict[str, int]]  # qid -> {doc_id: relevance}


def available_text_datasets() -> list[str]:
    if not BEIR_DIR.exists():
        return []
    return sorted(p.name for p in BEIR_DIR.iterdir() if (p / "corpus.jsonl").exists())


def load_beir(name: str, split: str = "test") -> BeirData:
    d = BEIR_DIR / name
    corpus: dict[str, str] = {}
    with (d / "corpus.jsonl").open() as f:
        for line in f:
            o = json.loads(line)
            title = o.get("title", "").strip()
            text = o.get("text", "").strip()
            corpus[o["_id"]] = f"{title}. {text}" if title else text

    queries: dict[str, str] = {}
    with (d / "queries.jsonl").open() as f:
        for line in f:
            o = json.loads(line)
            queries[o["_id"]] = o["text"]

    qrels: dict[str, dict[str, int]] = {}
    with (d / "qrels" / f"{split}.tsv").open() as f:
        reader = csv.reader(f, delimiter="\t")
        header = next(reader)  # "query-id  corpus-id  score"
        if header and header[0].lstrip("-").isdigit():
            # No header row — rewind by processing this row too.
            qid, did, score = header
            qrels.setdefault(qid, {})[did] = int(score)
        for qid, did, score in reader:
            qrels.setdefault(qid, {})[did] = int(score)

    # Keep only queries that have judgments for `split`.
    queries = {q: queries[q] for q in qrels if q in queries}
    return BeirData(name=name, corpus=corpus, queries=queries, qrels=qrels)


# --------------------------------------------------------------------------
# Smoke test
# --------------------------------------------------------------------------
if __name__ == "__main__":
    print("== vector datasets ==")
    for name in available_vector_datasets():
        ds = load_vectors(name, n=2000, nq=8)
        print(
            f"  {name:<24} dim={ds.dim:<4} metric={ds.metric:<6} corpus_len={ds.corpus_len} q_len={ds.query_len}"
        )

    print("\n== ground-truth sanity ==")
    # Computed GT (cohere subset, cosine).
    cohere = load_vectors("cohere-medium-1m-f32", n=2000, nq=8)
    gt = ground_truth(cohere, k=10)
    print(
        f"  cohere n=2000 nq=8 -> gt {gt.shape} dtype={gt.dtype}, q0 top3={gt[0, :3].tolist()}"
    )
    # Official GT (sift, full corpus, L2).
    sift = load_vectors("sift-1m", nq=8)
    gt_sift = ground_truth(sift, k=10)
    print(
        f"  sift full nq=8 (official) -> gt {gt_sift.shape}, q0 top3={gt_sift[0, :3].tolist()}"
    )

    print("\n== text datasets ==")
    for name in available_text_datasets():
        try:
            bd = load_beir(name)
            print(
                f"  {name:<18} docs={len(bd.corpus):<7} judged_queries={len(bd.queries):<5} qrels={sum(len(v) for v in bd.qrels.values())}"
            )
        except FileNotFoundError as e:
            print(f"  {name:<18} (no test split: {e})")
