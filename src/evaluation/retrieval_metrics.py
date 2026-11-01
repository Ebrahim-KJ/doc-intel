"""Ranking metrics for retrieval against graded relevance labels (qrels)."""

from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

Qrels = Dict[str, Dict[str, int]]   # query_id -> {doc_id: relevance}
Run = Dict[str, List[str]]          # query_id -> ranked doc_ids


def recall_at_k(ranked: List[str], relevant: Dict[str, int], k: int) -> float:
    rel = {d for d, g in relevant.items() if g > 0}
    if not rel:
        return float("nan")
    return len(rel.intersection(ranked[:k])) / len(rel)


def mrr_at_k(ranked: List[str], relevant: Dict[str, int], k: int) -> float:
    for i, d in enumerate(ranked[:k]):
        if relevant.get(d, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(ranked: List[str], relevant: Dict[str, int], k: int) -> float:
    dcg = sum((2 ** relevant.get(d, 0) - 1) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    ideal = sorted((g for g in relevant.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else float("nan")


def per_query(run: Run, qrels: Qrels, k: int = 10, recall_k: int = 100) -> Dict[str, np.ndarray]:
    qids = [q for q in qrels if q in run]
    return {
        f"ndcg@{k}": np.array([ndcg_at_k(run[q], qrels[q], k) for q in qids]),
        f"mrr@{k}": np.array([mrr_at_k(run[q], qrels[q], k) for q in qids]),
        f"recall@{k}": np.array([recall_at_k(run[q], qrels[q], k) for q in qids]),
        f"recall@{recall_k}": np.array([recall_at_k(run[q], qrels[q], recall_k) for q in qids]),
    }


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n_boot: int = 10_000, seed: int = 0) -> Dict[str, float]:
    """Mean of a - b over queries, with a 95% CI from resampling queries."""
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, d.size, size=(n_boot, d.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"diff": float(d.mean()), "lo": float(lo), "hi": float(hi), "p_le_0": float((means <= 0).mean())}
