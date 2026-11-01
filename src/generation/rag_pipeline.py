"""
rag_pipeline.py
---------------
Orchestrates the full RAG pipeline:

  Query → Multi-query expansion
        → Hybrid retrieval (FAISS + BM25) on all variants
        → RRF fusion
        → Cross-encoder reranking
        → Prompt assembly (context + conversation history)
        → Claude generation (streaming or batch)
        → Citation extraction
        → MLflow logging
"""

import os
import time
from typing import List, Dict, Any, Iterator, Union
import mlflow

from src.retrieval.vector_store   import get_store
from src.retrieval.bm25_retriever import get_bm25
from src.retrieval.hybrid_retriever import hybrid_search, reciprocal_rank_fusion
from src.retrieval.reranker       import rerank
from src.generation.multi_query   import expand_query
from src.generation.llm_client    import complete, stream as llm_stream
from src.generation.prompt_templates import build_system_prompt, extract_citations
from src.memory.conversation      import get_memory

DENSE_TOP_K  = int(os.getenv("DENSE_TOP_K", 20))
SPARSE_TOP_K = int(os.getenv("SPARSE_TOP_K", 20))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", 5))


def _retrieve(query: str) -> tuple[List[Dict], Dict[str, float]]:
    """
    Full retrieval pipeline with timing instrumentation.
    Returns (top_chunks, stage_latencies).
    """
    timings = {}
    store   = get_store()
    bm25    = get_bm25()

    # 1. Multi-query expansion
    t0      = time.time()
    queries = expand_query(query)
    timings["query_expansion_s"] = round(time.time() - t0, 3)

    # 2. Retrieve for each query variant, fuse results
    t0 = time.time()
    ranked_lists = []
    for q in queries:
        ranked_lists.append(store.search(q, top_k=DENSE_TOP_K))
        ranked_lists.append(bm25.search(q, top_k=SPARSE_TOP_K))

    # each variant's ranking is fused as its own list
    merged = reciprocal_rank_fusion(ranked_lists)
    timings["retrieval_s"] = round(time.time() - t0, 3)

    # 3. Cross-encoder reranking
    t0     = time.time()
    top_k  = rerank(query, merged[:40], top_k=RERANK_TOP_K)
    timings["rerank_s"] = round(time.time() - t0, 3)

    return top_k, timings


def query(
    user_query:  str,
    session_id:  str  = "default",
    streaming:   bool = False,
) -> Dict[str, Any]:
    """
    Full RAG query — non-streaming mode.
    Returns answer, citations, chunks, and stage timings.
    """
    memory = get_memory(session_id)
    t_total = time.time()

    # Retrieval
    top_chunks, timings = _retrieve(user_query)

    if not top_chunks:
        return {
            "answer":    "I could not find relevant information in the indexed documents.",
            "citations": [],
            "chunks":    [],
            "timings":   timings,
        }

    # Build prompt
    system    = build_system_prompt(top_chunks)
    messages  = memory.get_messages() + [{"role": "user", "content": user_query}]

    # Generate
    t0     = time.time()
    answer = complete(messages=messages, system=system)
    timings["generation_s"] = round(time.time() - t0, 3)
    timings["total_s"]      = round(time.time() - t_total, 3)

    # Update conversation memory
    memory.add_user(user_query)
    memory.add_assistant(answer)

    citations = extract_citations(top_chunks)

    # MLflow logging
    try:
        mlflow.set_experiment("doc-intel-queries")
        with mlflow.start_run(run_name="rag_query"):
            mlflow.log_param("query",        user_query[:100])
            mlflow.log_param("session_id",   session_id)
            mlflow.log_param("n_chunks",     len(top_chunks))
            mlflow.log_metrics(timings)
    except Exception:
        pass   # MLflow is optional

    return {
        "answer":    answer,
        "citations": citations,
        "chunks":    top_chunks,
        "timings":   timings,
    }


def query_stream(
    user_query: str,
    session_id: str = "default",
) -> Iterator[str]:
    """
    Streaming RAG query.
    Yields text chunks for use with st.write_stream().
    Side-effects: updates conversation memory after full response.
    """
    memory     = get_memory(session_id)
    top_chunks, _ = _retrieve(user_query)

    if not top_chunks:
        yield "I could not find relevant information in the indexed documents."
        return

    system   = build_system_prompt(top_chunks)
    messages = memory.get_messages() + [{"role": "user", "content": user_query}]

    full_response = []
    for chunk_text in llm_stream(messages=messages, system=system):
        full_response.append(chunk_text)
        yield chunk_text

    # Update memory after streaming completes
    memory.add_user(user_query)
    memory.add_assistant("".join(full_response))
