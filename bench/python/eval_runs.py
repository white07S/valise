# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""Score TREC run files with pytrec_eval against a BEIR dataset's qrels.

This is the *uniform* evaluator: the Rust bench writes `valise.<ds>.run` and
`tantivy.<ds>.run`, and these get scored by the exact same `pytrec_eval`
the Python text runners use — so Valise / tantivy / bm25s / pyserini nDCG are
directly comparable rather than computed by three different metric codes.

    uv run python eval_runs.py scifact results/runs/valise.scifact.run \
        results/runs/tantivy.scifact.run
"""

from __future__ import annotations

import sys
from pathlib import Path

import valise_data as D
from metrics import text_metrics


def read_trec_run(path: str | Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        qid, _q0, docid, _rank, score, _tag = parts[:6]
        run.setdefault(qid, {})[docid] = float(score)
    return run


def main() -> None:
    if len(sys.argv) < 3:
        print("usage: eval_runs.py <dataset> <run_file> [run_file ...]")
        raise SystemExit(2)
    dataset, run_files = sys.argv[1], sys.argv[2:]
    bd = D.load_beir(dataset)
    print(f"[eval] {dataset}: {len(bd.qrels)} judged queries (pytrec_eval)\n")
    print(f"{'run':<22}{'nDCG@10':>9}{'MAP':>8}{'MRR':>8}{'R@100':>8}{'queries':>9}")
    for rf in run_files:
        run = read_trec_run(rf)
        m = text_metrics(run, bd.qrels)
        print(
            f"{Path(rf).stem:<22}{m['ndcg_at_10']:>9.4f}{m['map']:>8.4f}"
            f"{m['mrr']:>8.4f}{m['recall_at_100']:>8.4f}{len(run):>9}"
        )


if __name__ == "__main__":
    main()
