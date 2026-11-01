"""
embeddings.py
-------------
Sentence-transformer embedding wrapper with caching.
"""

import os
import numpy as np
from typing import List
from functools import lru_cache

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
BATCH_SIZE = 64


@lru_cache(maxsize=1)
def _get_model():
    from sentence_transformers import SentenceTransformer
    print(f"📦 Loading embedding model: {MODEL_NAME}")
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: List[str], batch_size: int = BATCH_SIZE) -> np.ndarray:
    """Embed a list of texts. Returns (N, dim) float32 array."""
    model = _get_model()
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=len(texts) > 100,
        convert_to_numpy=True,
        normalize_embeddings=False,
    )
    return embeddings.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string."""
    return embed_texts([query])[0]
