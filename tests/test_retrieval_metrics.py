import math

import numpy as np
import pytest

from src.evaluation.retrieval_metrics import mrr_at_k, ndcg_at_k, paired_bootstrap, recall_at_k
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion

REL = {"a": 2, "b": 1, "c": 0}


def test_recall_and_mrr():
    assert recall_at_k(["x", "b", "a"], REL, 2) == 0.5
    assert recall_at_k(["x", "b", "a"], REL, 3) == 1.0
    assert mrr_at_k(["x", "y", "b"], REL, 10) == pytest.approx(1 / 3)
    assert mrr_at_k(["x", "y", "b"], REL, 2) == 0.0


def test_ndcg_by_hand():
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert ndcg_at_k(["b", "a"], REL, 10) == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b"], REL, 10) == pytest.approx(1.0)


def test_rrf_rewards_agreement():
    a = [{"chunk_id": i, "text": i} for i in ["x", "y", "z"]]
    b = [{"chunk_id": i, "text": i} for i in ["y", "w", "x"]]
    fused = [c["chunk_id"] for c in reciprocal_rank_fusion([a, b], k=60)]
    assert fused[0] == "y"
    assert set(fused) == {"x", "y", "z", "w"}


def test_paired_bootstrap_sign():
    rng = np.random.default_rng(0)
    a = rng.normal(0.6, 0.2, 300)
    b = a - 0.05 + rng.normal(0, 0.05, 300)
    r = paired_bootstrap(a, b, n_boot=2000)
    assert r["lo"] > 0 and r["diff"] == pytest.approx(0.05, abs=0.01)
