"""
reranker.py
-----------
Cross-encoder reranking using ms-marco-MiniLM-L-6-v2.

Unlike bi-encoders (used in dense retrieval), cross-encoders
jointly encode the query and each candidate chunk together,
producing a much more accurate relevance score at the cost of
being too slow to run over the full corpus (hence the two-stage design).

Stage 1: Bi-encoder retrieval narrows from N → ~40 candidates (fast)
Stage 2: Cross-encoder reranking narrows from ~40 → top-5 (accurate)
"""

import os
from typing import List, Dict, Any
from functools import lru_cache

RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
RERANK_TOP_K   = int(os.getenv("RERANK_TOP_K", 5))


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers.cross_encoder import CrossEncoder
    print(f"Loading cross-encoder reranker: {RERANKER_MODEL}")
    return CrossEncoder(RERANKER_MODEL)


def rerank(
    query:      str,
    candidates: List[Dict[str, Any]],
    top_k:      int = RERANK_TOP_K,
) -> List[Dict[str, Any]]:
    """
    Score each (query, chunk) pair with the cross-encoder and
    return the top_k highest-scoring chunks.
    """
    if not candidates:
        return []

    reranker = _get_reranker()

    pairs  = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(candidates, scores),
        key=lambda x: x[1],
        reverse=True
    )

    results = []
    for chunk, score in ranked[:top_k]:
        c = dict(chunk)
        c["rerank_score"]     = float(score)
        c["retrieval_mode"]   = "reranked"
        results.append(c)

    print(f"  Cross-encoder reranked {len(candidates)} → top {len(results)}")
    return results
