"""
pipeline.py
-----------
Full document ingestion orchestration:
  Load → Chunk → Embed → Index (FAISS + BM25) → Persist
"""

import os
import time
from typing import List, Dict, Any

from src.ingestion.document_loader import load_document
from src.ingestion.chunking        import chunk_documents
from src.retrieval.vector_store    import get_store
from src.retrieval.bm25_retriever  import get_bm25

CHUNK_STRATEGY = os.getenv("CHUNK_STRATEGY", "semantic")


def ingest(source: str, strategy: str = CHUNK_STRATEGY) -> Dict[str, Any]:
    """
    Ingest a single document (path or URL) into both indices.
    Returns ingestion stats.
    """
    t0 = time.time()
    print(f"\nIngesting: {source}")

    pages  = load_document(source)
    chunks = chunk_documents(pages, strategy=strategy)

    store = get_store()
    bm25  = get_bm25()

    store.add(chunks)
    bm25.add(chunks)

    store.save()
    bm25.save()

    elapsed = round(time.time() - t0, 2)
    stats   = {
        "source":   source,
        "pages":    len(pages),
        "chunks":   len(chunks),
        "strategy": strategy,
        "elapsed_s": elapsed,
    }
    print(f"Ingested in {elapsed}s — {len(chunks)} chunks indexed")
    return stats


def ingest_many(sources: List[str]) -> List[Dict[str, Any]]:
    return [ingest(s) for s in sources]


def load_indices() -> bool:
    """Load persisted indices at startup. Returns True if found."""
    store   = get_store()
    bm25    = get_bm25()
    loaded  = store.load() and bm25.load()
    if not loaded:
        print("ℹ️  No existing index found — ingest documents to begin.")
    return loaded


if __name__ == "__main__":
    # Quick test: ingest a Wikipedia article
    stats = ingest("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    print(stats)
