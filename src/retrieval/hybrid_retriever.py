"""
hybrid_retriever.py
-------------------
Merges dense (FAISS) and sparse (BM25) results using
Reciprocal Rank Fusion (RRF).

RRF formula:  score(d) = Σ 1 / (k + rank_i(d))
where k=60 is a constant that dampens the influence of very high ranks.

Chunks appearing in both ranked lists get a compounded score,
rewarding those that are semantically AND lexically relevant.
"""

from typing import List, Dict, Any


def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60
) -> List[Dict[str, Any]]:
    """
    Merge multiple ranked lists into one using RRF.
    Each list is a list of chunk dicts with a 'chunk_id' field.
    """
    rrf_scores: Dict[str, float] = {}
    chunk_map:  Dict[str, Dict]  = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list):
            cid = chunk["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            if cid not in chunk_map:
                chunk_map[cid] = chunk

    merged = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    result = []
    for cid, score in merged:
        chunk = dict(chunk_map[cid])
        chunk["rrf_score"]      = score
        chunk["retrieval_mode"] = "hybrid_rrf"
        result.append(chunk)

    return result


def hybrid_search(
    query: str,
    faiss_store,
    bm25_retriever,
    dense_top_k:  int = 20,
    sparse_top_k: int = 20,
) -> List[Dict[str, Any]]:
    """
    Run dense + sparse retrieval in parallel and fuse results with RRF.
    Returns merged list sorted by RRF score.
    """
    dense_results  = faiss_store.search(query, top_k=dense_top_k)
    sparse_results = bm25_retriever.search(query, top_k=sparse_top_k)

    merged = reciprocal_rank_fusion([dense_results, sparse_results])

    print(f"  Hybrid retrieval: dense={len(dense_results)}, "
          f"sparse={len(sparse_results)}, merged={len(merged)}")
    return merged
