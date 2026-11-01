"""Retrieval benchmark: BM25 vs dense vs hybrid (RRF) vs cross-encoder reranking.

Runs on BEIR datasets with human relevance labels, so every number is
measured against ground truth rather than an LLM's opinion.

    python benchmarks/retrieval_benchmark.py scifact fiqa
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation import beir  # noqa: E402
from src.evaluation.retrieval_metrics import paired_bootstrap, per_query  # noqa: E402
from src.retrieval.bm25_retriever import BM25Retriever  # noqa: E402
from src.retrieval.hybrid_retriever import reciprocal_rank_fusion  # noqa: E402
from src.retrieval.reranker import _get_reranker  # noqa: E402
from src.retrieval.vector_store import FAISSVectorStore  # noqa: E402

OUT = ROOT / "results"
DEPTH = 100
RERANK_DEPTH = 50


def ids(hits):
    return [h["chunk_id"] for h in hits]


def rerank_all(queries, candidates, depth):
    model = _get_reranker()
    out = []
    for q, cands in zip(queries, candidates):
        cands = cands[:depth]
        scores = model.predict([(q, c["text"]) for c in cands], batch_size=64)
        out.append([c for _, c in sorted(zip(scores, cands), key=lambda x: -x[0])])
    return out


def run(name: str) -> dict:
    corpus, queries, qrels = beir.load(name)
    chunks = beir.as_chunks(corpus, name)
    qids = list(queries)
    qtext = [queries[q] for q in qids]
    print(f"{name}: {len(chunks):,} documents, {len(qids)} test queries")

    timing, runs = {}, {}

    bm25 = BM25Retriever()
    bm25.build(chunks)
    t = time.perf_counter()
    sparse = [bm25.search(q, top_k=DEPTH) for q in qtext]
    timing["BM25"] = (time.perf_counter() - t) / len(qids)

    store = FAISSVectorStore()
    t = time.perf_counter()
    store.build(chunks)
    index_s = time.perf_counter() - t
    t = time.perf_counter()
    dense = store.search_many(qtext, top_k=DEPTH)
    timing["Dense"] = (time.perf_counter() - t) / len(qids)

    hybrid = [reciprocal_rank_fusion([d, s]) for d, s in zip(dense, sparse)]
    timing["Hybrid (RRF)"] = timing["BM25"] + timing["Dense"]

    t = time.perf_counter()
    hybrid_rr = rerank_all(qtext, hybrid, RERANK_DEPTH)
    timing["Hybrid + rerank"] = timing["Hybrid (RRF)"] + (time.perf_counter() - t) / len(qids)
    t = time.perf_counter()
    dense_rr = rerank_all(qtext, dense, RERANK_DEPTH)
    timing["Dense + rerank"] = timing["Dense"] + (time.perf_counter() - t) / len(qids)

    for label, hits in [("BM25", sparse), ("Dense", dense), ("Hybrid (RRF)", hybrid),
                        ("Dense + rerank", dense_rr), ("Hybrid + rerank", hybrid_rr)]:
        runs[label] = {q: ids(h) for q, h in zip(qids, hits)}

    scores = {label: per_query(r, qrels) for label, r in runs.items()}
    table = {label: {m: float(np.nanmean(v)) for m, v in s.items()} | {"ms_per_query": timing[label] * 1000}
             for label, s in scores.items()}
    m = "ndcg@10"
    comparisons = {
        "Dense vs BM25": paired_bootstrap(scores["Dense"][m], scores["BM25"][m]),
        "Hybrid vs Dense": paired_bootstrap(scores["Hybrid (RRF)"][m], scores["Dense"][m]),
        "Hybrid vs BM25": paired_bootstrap(scores["Hybrid (RRF)"][m], scores["BM25"][m]),
        "Hybrid+rerank vs Hybrid": paired_bootstrap(scores["Hybrid + rerank"][m], scores["Hybrid (RRF)"][m]),
        "Hybrid+rerank vs Dense+rerank": paired_bootstrap(scores["Hybrid + rerank"][m], scores["Dense + rerank"][m]),
    }
    res = {"dataset": name, "documents": len(chunks), "queries": len(qids), "index_build_s": index_s,
           "table": table, "comparisons": comparisons,
           "per_query_ndcg10": {label: s[m].tolist() for label, s in scores.items()}}
    OUT.mkdir(exist_ok=True)
    (OUT / f"retrieval_{name}.json").write_text(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    for ds in sys.argv[1:] or ["scifact"]:
        r = run(ds)
        for label, row in r["table"].items():
            print(f"  {label:<18} nDCG@10 {row['ndcg@10']:.3f}  R@100 {row['recall@100']:.3f}  "
                  f"{row['ms_per_query']:.1f} ms/query")
        for k, v in r["comparisons"].items():
            print(f"  {k:<30} {v['diff']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}]")
