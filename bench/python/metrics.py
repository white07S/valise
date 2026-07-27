"""Quality metrics: vector recall@k vs exact GT, text IR metrics vs qrels."""

from __future__ import annotations

import numpy as np


def recall_at_k(retrieved: np.ndarray, gt: np.ndarray, k: int) -> float:
    """Mean Recall@k = |retrieved[:k] ∩ gt[:k]| / k, over queries.

    `retrieved` (nq, K) and `gt` (nq, Kgt) are corpus-index arrays; both
    must have at least `k` columns.
    """
    nq = retrieved.shape[0]
    if k > retrieved.shape[1] or k > gt.shape[1]:
        raise ValueError(
            f"recall@{k}: need >= {k} cols (got retrieved={retrieved.shape[1]}, gt={gt.shape[1]})"
        )
    hit = 0
    for i in range(nq):
        truth = set(gt[i, :k].tolist())
        truth.discard(-1)
        if not truth:
            continue
        got = set(retrieved[i, :k].tolist())
        hit += len(truth & got)
    denom = sum(min(k, len(set(gt[i, :k].tolist()) - {-1})) for i in range(nq))
    return hit / denom if denom else 0.0


def text_metrics(
    run: dict[str, dict[str, float]], qrels: dict[str, dict[str, int]]
) -> dict[str, float]:
    """nDCG@10 / MAP / MRR / Recall@100 via pytrec_eval (TREC-correct).

    `run[qid][doc_id] = score`; only queries present in `qrels` are scored.
    """
    import pytrec_eval

    measures = {"ndcg_cut.10", "map", "recip_rank", "recall.100"}
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    scores = evaluator.evaluate(run)
    if not scores:
        return {"ndcg_at_10": 0.0, "map": 0.0, "mrr": 0.0, "recall_at_100": 0.0}

    def mean(key: str) -> float:
        return float(np.mean([s[key] for s in scores.values()]))

    return {
        "ndcg_at_10": mean("ndcg_cut_10"),
        "map": mean("map"),
        "mrr": mean("recip_rank"),
        "recall_at_100": mean("recall_100"),
    }
