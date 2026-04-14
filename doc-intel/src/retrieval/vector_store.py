"""
vector_store.py
---------------
Manages the FAISS dense vector index.
Wraps build, persist, load, and search operations with full
chunk metadata retrieval.
"""

import os
import json
import numpy as np
import faiss
import joblib
from typing import List, Dict, Any, Tuple

from src.retrieval.embeddings import embed_texts, embed_query

INDEX_DIR    = os.path.join(os.path.dirname(__file__), "../../data/indices")
INDEX_PATH   = os.path.join(INDEX_DIR, "faiss.index")
METADATA_PATH = os.path.join(INDEX_DIR, "metadata.json")


class FAISSVectorStore:
    """
    FAISS flat index (cosine similarity via L2-normalised vectors).
    Stores parallel metadata list for chunk retrieval.
    """

    def __init__(self):
        self.index    = None
        self.metadata = []   # List of chunk dicts (text, source, page, etc.)
        self.dim      = None

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """Embed all chunks and build the FAISS index."""
        print(f"🔨 Building FAISS index over {len(chunks)} chunks...")
        texts      = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)         # (N, dim) float32

        # L2-normalise for cosine similarity via inner product
        faiss.normalize_L2(embeddings)

        self.dim      = embeddings.shape[1]
        self.index    = faiss.IndexFlatIP(self.dim)   # Inner product on normalised = cosine
        self.index.add(embeddings)
        self.metadata = [
            {k: c[k] for k in ("text", "source", "page", "doc_id", "chunk_id", "chunk_idx")}
            for c in chunks
        ]
        print(f"✅ FAISS index built: {self.index.ntotal} vectors, dim={self.dim}")

    def add(self, chunks: List[Dict[str, Any]]) -> None:
        """Incrementally add new chunks to an existing index."""
        if self.index is None:
            return self.build(chunks)

        texts      = [c["text"] for c in chunks]
        embeddings = embed_texts(texts)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.metadata.extend([
            {k: c[k] for k in ("text", "source", "page", "doc_id", "chunk_id", "chunk_idx")}
            for c in chunks
        ])
        print(f"✅ Added {len(chunks)} chunks — total: {self.index.ntotal}")

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Return top_k most similar chunks with scores."""
        if self.index is None or self.index.ntotal == 0:
            return []

        q_emb = embed_query(query).reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(q_emb)

        scores, indices = self.index.search(q_emb, min(top_k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"]          = float(score)
            chunk["retrieval_mode"] = "dense"
            results.append(chunk)

        return results

    def save(self) -> None:
        """Persist index and metadata to disk."""
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)
        print(f"✅ Saved FAISS index → {INDEX_PATH}")

    def load(self) -> bool:
        """Load persisted index. Returns True if successful."""
        if not (os.path.exists(INDEX_PATH) and os.path.exists(METADATA_PATH)):
            return False
        self.index = faiss.read_index(INDEX_PATH)
        with open(METADATA_PATH) as f:
            self.metadata = json.load(f)
        self.dim = self.index.d
        print(f"✅ Loaded FAISS index: {self.index.ntotal} vectors")
        return True

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0


# Singleton instance for shared use across the app
_store = FAISSVectorStore()

def get_store() -> FAISSVectorStore:
    return _store
