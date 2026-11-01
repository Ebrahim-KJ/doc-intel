"""
bm25_retriever.py
-----------------
BM25 sparse retrieval — complements dense search by handling
exact keyword matches and rare terminology that embeddings may miss.
"""

import os
import re
import json
import joblib
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi

INDEX_DIR   = os.path.join(os.path.dirname(__file__), "../../data/indices")
BM25_PATH   = os.path.join(INDEX_DIR, "bm25.joblib")
BM25_META_PATH = os.path.join(INDEX_DIR, "bm25_meta.json")


def _tokenise(text: str) -> List[str]:
    """Lowercase, remove punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


class BM25Retriever:

    def __init__(self):
        self.bm25     = None
        self.metadata = []

    def build(self, chunks: List[Dict[str, Any]]) -> None:
        """Tokenise all chunks and fit BM25."""
        print(f"Building BM25 index over {len(chunks)} chunks...")
        tokenised     = [_tokenise(c["text"]) for c in chunks]
        self.bm25     = BM25Okapi(tokenised)
        self.metadata = [
            {k: c[k] for k in ("text", "source", "page", "doc_id", "chunk_id", "chunk_idx")}
            for c in chunks
        ]
        print(f"BM25 index built: {len(self.metadata)} documents")

    def add(self, chunks: List[Dict[str, Any]]) -> None:
        """
        BM25Okapi doesn't support incremental adds natively.
        Rebuild the index with the full corpus.
        """
        all_chunks = list(self.metadata) + chunks
        self.build(all_chunks)

    def search(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """Return top_k chunks ranked by BM25 score."""
        if self.bm25 is None:
            return []

        tokens = _tokenise(query)
        scores = self.bm25.get_scores(tokens)

        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] <= 0:
                continue
            chunk = dict(self.metadata[idx])
            chunk["score"]          = float(scores[idx])
            chunk["retrieval_mode"] = "sparse"
            results.append(chunk)

        return results

    def save(self) -> None:
        os.makedirs(INDEX_DIR, exist_ok=True)
        joblib.dump(self.bm25, BM25_PATH)
        with open(BM25_META_PATH, "w") as f:
            json.dump(self.metadata, f)
        print(f"Saved BM25 index → {BM25_PATH}")

    def load(self) -> bool:
        if not (os.path.exists(BM25_PATH) and os.path.exists(BM25_META_PATH)):
            return False
        self.bm25 = joblib.load(BM25_PATH)
        with open(BM25_META_PATH) as f:
            self.metadata = json.load(f)
        print(f"Loaded BM25 index: {len(self.metadata)} documents")
        return True

    @property
    def size(self) -> int:
        return len(self.metadata)


_bm25 = BM25Retriever()

def get_bm25() -> BM25Retriever:
    return _bm25
