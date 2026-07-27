# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Download and convert the benchmark datasets.

The benchmark corpora are large and regenerable, so they are not in the
repository. This fetches them and writes them in the layout the harnesses
expect, so the numbers in REPRODUCE.md can be re-run from a clean clone.

    python3 bench/prep_data.py --list
    python3 bench/prep_data.py scifact          # 3 MB, no dependencies
    python3 bench/prep_data.py sift-1m          # 501 MB, needs h5py

Text datasets need nothing but the standard library. Vector datasets are
distributed as HDF5 and need `h5py` and `numpy`:

    pip install h5py numpy

Everything is skipped if it is already present; pass --force to redo it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
BEIR_DIR = HERE / "beir-data"
VEC_DIR = HERE / "datasets"

# BEIR text corpora. The zip already contains corpus.jsonl, queries.jsonl
# and qrels/test.tsv, which is exactly the layout the harness wants.
TEXT = {
    "scifact": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip",
        "size": "3 MB",
        "note": "5 183 docs, 300 evaluated queries. The default text benchmark.",
    },
    "nfcorpus": {
        "url": "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip",
        "size": "3 MB",
        "note": "3 633 docs. Small; useful for a fast smoke run.",
    },
}

# ann-benchmarks HDF5 bundles: `train` is the corpus, `test` the queries,
# and `neighbors` the exact top-100 ground truth. Names must keep their
# prefix — bench/python/valise_data.py::metric_for derives the metric from
# it (sift/gist -> l2, glove -> cosine).
VECTOR = {
    "sift-1m": {
        "url": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
        "size": "501 MB",
        "dim": 128,
        "note": "Classic texmex SIFT, 1M x 128, L2.",
    },
    "gist-1m": {
        "url": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "size": "3.6 GB",
        "dim": 960,
        "note": "texmex GIST, 1M x 960, L2. Large download.",
    },
    "glove-100": {
        "url": "http://ann-benchmarks.com/glove-100-angular.hdf5",
        "size": "463 MB",
        "dim": 100,
        "note": "GloVe word vectors, 1.18M x 100, angular (= cosine).",
    },
}

# Datasets the benchmark supports but which cannot be fetched without
# credentials or a very large transfer. Documented rather than automated.
MANUAL = {
    "cohere-medium-1m-f32": (
        "1M x 768 cosine. Build from the HuggingFace dataset "
        "`Cohere/wikipedia-22-12-en-embeddings` (gated: accept the terms, "
        "then `huggingface-cli login`). Write corpus.f32 / queries.f32 / "
        "meta.json in the layout described by `--layout`."
    ),
    "openai-medium-500k": (
        "500k x 1536 cosine. Build from any OpenAI text-embedding-3 dump "
        "in the same layout. Not redistributable."
    ),
}

LAYOUT = """\
Vector dataset layout (bench/datasets/<name>/):

  corpus.f32   little-endian f32, row-major, shape (corpus_len, dim)
  queries.f32  little-endian f32, row-major, shape (query_len, dim)
  gt.u32       optional; little-endian u32, shape (query_len, gt_k),
               exact nearest-neighbour indices into the corpus
  meta.json    {"dim": int, "corpus_len": int, "query_len": int,
                "gt_k": int}          # gt_k only if gt.u32 is present

The name matters: bench/python/valise_data.py::metric_for maps the prefix
to a metric (sift/gist -> l2, cohere/openai/glove/mpnet -> cosine) and
refuses to guess for anything else.
"""


def _download(url: str, dest: Path) -> None:
    """Stream to a temp file next to `dest`, then rename.

    Downloading straight to `dest` would leave a truncated file looking
    complete if the transfer died, and the next run would skip it.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  fetching {url}")
    tmp = dest.with_suffix(dest.suffix + ".partial")
    # Some of these hosts reject urllib's default "Python-urllib/3.x" agent
    # with a 403 while serving the same URL fine to curl, so send a normal
    # one. Do not remove this: a HEAD check will pass without it.
    req = urllib.request.Request(url, headers={"User-Agent": "valise-bench-prep/1.0"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - fixed URLs above
        total = int(resp.headers.get("content-length") or 0)
        done = 0
        with tmp.open("wb") as out:
            while chunk := resp.read(1 << 20):
                out.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done >> 20} / {total >> 20} MiB ({pct:.0f}%)", end="")
        print()
    tmp.rename(dest)


def fetch_text(name: str, force: bool) -> None:
    spec = TEXT[name]
    out = BEIR_DIR / name
    if (out / "corpus.jsonl").exists() and not force:
        print(f"{name}: already present at {out}")
        return
    print(f"{name}: {spec['size']} — {spec['note']}")
    with tempfile.TemporaryDirectory() as td:
        archive = Path(td) / f"{name}.zip"
        _download(spec["url"], archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(td)
        extracted = Path(td) / name
        if not (extracted / "corpus.jsonl").exists():
            sys.exit(f"{name}: archive layout unexpected; no {extracted}/corpus.jsonl")
        if out.exists():
            shutil.rmtree(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(extracted), str(out))
    print(f"{name}: ready at {out}")


def fetch_vector(name: str, force: bool) -> None:
    try:
        import h5py
        import numpy as np
    except ImportError:
        sys.exit(
            f"{name} is distributed as HDF5 and needs h5py + numpy:\n"
            "    pip install h5py numpy"
        )

    spec = VECTOR[name]
    out = VEC_DIR / name
    if (out / "meta.json").exists() and not force:
        print(f"{name}: already present at {out}")
        return
    print(f"{name}: {spec['size']} — {spec['note']}")
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as td:
        h5_path = Path(td) / f"{name}.hdf5"
        _download(spec["url"], h5_path)
        with h5py.File(h5_path, "r") as h5:
            missing = [k for k in ("train", "test") if k not in h5]
            if missing:
                sys.exit(
                    f"{name}: HDF5 is missing {missing}; found {list(h5.keys())}. "
                    "The ann-benchmarks layout may have changed."
                )
            corpus = np.asarray(h5["train"], dtype="<f4")
            queries = np.asarray(h5["test"], dtype="<f4")
            gt = np.asarray(h5["neighbors"], dtype="<u4") if "neighbors" in h5 else None

        if corpus.shape[1] != spec["dim"]:
            sys.exit(f"{name}: expected dim {spec['dim']}, got {corpus.shape[1]}")

        corpus.tofile(out / "corpus.f32")
        queries.tofile(out / "queries.f32")
        meta = {
            "dim": int(corpus.shape[1]),
            "corpus_len": int(corpus.shape[0]),
            "query_len": int(queries.shape[0]),
        }
        if gt is not None:
            gt.tofile(out / "gt.u32")
            meta["gt_k"] = int(gt.shape[1])
        (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"{name}: ready at {out}  ({meta['corpus_len']} x {meta['dim']})")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Download and convert Valise benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=LAYOUT,
    )
    ap.add_argument("datasets", nargs="*", help="dataset names, or 'all-small'")
    ap.add_argument("--list", action="store_true", help="show available datasets")
    ap.add_argument("--layout", action="store_true", help="show the on-disk layout")
    ap.add_argument("--force", action="store_true", help="re-fetch even if present")
    args = ap.parse_args()

    if args.layout:
        print(LAYOUT)
        return

    if args.list or not args.datasets:
        print("Text (standard library only):")
        for n, s in TEXT.items():
            print(f"  {n:<24} {s['size']:>8}   {s['note']}")
        print("\nVector (needs h5py + numpy):")
        for n, s in VECTOR.items():
            print(f"  {n:<24} {s['size']:>8}   {s['note']}")
        print("\nShortcut:")
        print(f"  {'all-small':<24} {'504 MB':>8}   scifact + sift-1m")
        print("\nManual (not redistributable / credentialed):")
        for n, why in MANUAL.items():
            print(f"  {n}\n      {why}")
        if not args.datasets:
            print("\nNothing fetched. Name a dataset, e.g. `prep_data.py scifact`.")
        return

    requested: list[str] = []
    for name in args.datasets:
        if name == "all-small":
            requested += ["scifact", "sift-1m"]
        else:
            requested.append(name)

    for name in requested:
        if name in TEXT:
            fetch_text(name, args.force)
        elif name in VECTOR:
            fetch_vector(name, args.force)
        elif name in MANUAL:
            sys.exit(f"{name} cannot be fetched automatically.\n  {MANUAL[name]}")
        else:
            sys.exit(f"unknown dataset {name!r}; run --list to see the options")


if __name__ == "__main__":
    main()
