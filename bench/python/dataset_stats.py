# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Per-dataset vector statistics for the paper's dataset table.

For every vector dataset under ``bench/datasets/`` (anything with a
``meta.json``), computes on a fixed-seed sample of up to 50k corpus rows:

* dim (stored) and native_dim (pre-padding, from meta.json when present)
* corpus_len / query_len
* norm mean and CV (std/mean) — "is this model family unit-norm?"
* top-1 / top-10 PCA energy fraction (anisotropy; sampled PCA)
* fraction of negative component values

Stats are computed over the *native* dimensions only, so zero-padding
(e.g. glove-100's 100 -> 128) does not dilute them. Padding contributes
zero variance, so the PCA energy fractions are identical either way; the
negative-fraction and norm stats are what the native slice protects.

Usage::

    uv run python dataset_stats.py            # all datasets
    uv run python dataset_stats.py glove-100  # subset

Writes ``bench/results/paper/dataset_stats.json``.
"""

from __future__ import annotations

import json
import sys
from datetime import date

import numpy as np

import valise_data

SAMPLE_ROWS = 50_000
SEED = 0
OUT = valise_data.BENCH_DIR / "results" / "paper" / "dataset_stats.json"


def dataset_stats(name: str) -> dict:
    meta = json.loads((valise_data.VEC_DIR / name / "meta.json").read_text())
    ds = valise_data.load_vectors(name)
    native_dim = int(meta.get("native_dim", ds.dim))

    rng = np.random.default_rng(SEED)
    n_sample = min(SAMPLE_ROWS, ds.corpus_len)
    idx = np.sort(rng.choice(ds.corpus_len, size=n_sample, replace=False))
    # f64 throughout: gist-1m values are tiny (~1e-2) and 50k x 1536 fits easily.
    x = np.asarray(ds.corpus[idx, :native_dim], dtype=np.float64)

    norms = np.linalg.norm(x, axis=1)
    norm_mean = float(norms.mean())
    norm_cv = float(norms.std() / norm_mean) if norm_mean > 0 else float("nan")

    centered = x - x.mean(axis=0)
    cov = centered.T @ centered / (n_sample - 1)
    eig = np.linalg.eigvalsh(cov)[::-1]  # descending
    total = float(eig.sum())
    pca_top1 = float(eig[0] / total)
    pca_top10 = float(eig[:10].sum() / total)

    frac_negative = float((x < 0).mean())

    return {
        "dim": ds.dim,
        "native_dim": native_dim,
        "padded": bool(meta.get("padded", False)),
        "corpus_len": ds.corpus_len,
        "query_len": ds.query_len,
        "metric": ds.metric,
        "sample_rows": n_sample,
        "norm_mean": round(norm_mean, 6),
        "norm_cv": round(norm_cv, 6),
        "pca_energy_top1": round(pca_top1, 6),
        "pca_energy_top10": round(pca_top10, 6),
        "frac_negative": round(frac_negative, 6),
    }


def main() -> None:
    names = sys.argv[1:] or valise_data.available_vector_datasets()
    out = {
        "generated": date.today().isoformat(),
        "sample_rows": SAMPLE_ROWS,
        "seed": SEED,
        "note": (
            "stats over native (pre-padding) dims on a fixed-seed sample of "
            "corpus rows; pca_energy_topK = top-K eigenvalue mass fraction of "
            "the sample covariance"
        ),
        "datasets": {},
    }
    for name in names:
        stats = dataset_stats(name)
        out["datasets"][name] = stats
        print(
            f"{name:<24} d={stats['native_dim']}->{stats['dim']:<5}"
            f" n={stats['corpus_len']:<8} norm={stats['norm_mean']:.4f}"
            f" cv={stats['norm_cv']:.4f} pca1={stats['pca_energy_top1']:.3f}"
            f" pca10={stats['pca_energy_top10']:.3f}"
            f" neg={stats['frac_negative']:.3f}"
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
