"""
chunking.py
-----------
Two chunking strategies:
  1. Fixed-size with overlap  — simple, consistent chunk length
  2. Semantic chunker         — splits at sentence boundaries, groups by
                                embedding similarity (avoids mid-sentence cuts)

Each chunk carries full provenance metadata.
"""

import re
import os
from typing import List, Dict, Any

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 512))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 64))


# FIXED-SIZE CHUNKER

def chunk_fixed(pages: List[Dict[str, Any]],
                chunk_size: int = CHUNK_SIZE,
                overlap: int    = CHUNK_OVERLAP) -> List[Dict[str, Any]]:
    """
    Split page text into fixed-size character windows with overlap.
    Fast and predictable — good baseline.
    """
    chunks = []
    for page in pages:
        text  = page["text"]
        start = 0
        idx   = 0
        while start < len(text):
            end   = min(start + chunk_size, len(text))
            chunk = text[start:end].strip()
            if len(chunk) > 30:
                chunks.append({
                    **{k: page[k] for k in ("source", "page", "doc_id")},
                    "text":       chunk,
                    "chunk_id":   f"{page['doc_id']}_p{page['page']}_c{idx}",
                    "chunk_idx":  idx,
                    "strategy":   "fixed",
                })
                idx += 1
            start += chunk_size - overlap

    print(f"Fixed chunker: {len(pages)} pages → {len(chunks)} chunks "
          f"(size={chunk_size}, overlap={overlap})")
    return chunks


# SEMANTIC CHUNKER

def _split_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation heuristics."""
    pattern = r'(?<=[.!?])\s+(?=[A-Z])'
    sentences = re.split(pattern, text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def chunk_semantic(pages: List[Dict[str, Any]],
                   max_tokens: int = CHUNK_SIZE,
                   similarity_threshold: float = 0.5) -> List[Dict[str, Any]]:
    """
    Sentence-aware chunking:
      1. Split each page into sentences
      2. Group consecutive sentences into chunks ≤ max_tokens chars
      3. Start a new chunk when adding a sentence would exceed the limit
         OR when the sentence is topically dissimilar (cosine sim < threshold)

    Falls back to fixed chunking if sentence-transformers not available.
    """
    try:
        from sentence_transformers import SentenceTransformer
        import numpy as np
        model = SentenceTransformer("all-MiniLM-L6-v2")
        use_similarity = True
    except ImportError:
        use_similarity = False

    chunks = []
    for page in pages:
        sentences   = _split_sentences(page["text"])
        if not sentences:
            continue

        if use_similarity:
            embeddings = model.encode(sentences, show_progress_bar=False)

        current_sents  = [sentences[0]]
        current_len    = len(sentences[0])
        chunk_idx      = 0

        for i in range(1, len(sentences)):
            sent     = sentences[i]
            sent_len = len(sent)

            # Check similarity with previous sentence
            if use_similarity:
                import numpy as np
                sim = float(np.dot(embeddings[i], embeddings[i-1]) /
                            (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1]) + 1e-9))
                topic_break = sim < similarity_threshold
            else:
                topic_break = False

            if (current_len + sent_len > max_tokens) or topic_break:
                # Flush current chunk
                chunk_text = " ".join(current_sents).strip()
                if len(chunk_text) > 30:
                    chunks.append({
                        **{k: page[k] for k in ("source", "page", "doc_id")},
                        "text":      chunk_text,
                        "chunk_id":  f"{page['doc_id']}_p{page['page']}_c{chunk_idx}",
                        "chunk_idx": chunk_idx,
                        "strategy":  "semantic",
                    })
                    chunk_idx += 1

                # Carry overlap: keep last sentence in new chunk
                current_sents = [current_sents[-1], sent] if current_sents else [sent]
                current_len   = sum(len(s) for s in current_sents)
            else:
                current_sents.append(sent)
                current_len += sent_len

        # Flush final chunk
        if current_sents:
            chunk_text = " ".join(current_sents).strip()
            if len(chunk_text) > 30:
                chunks.append({
                    **{k: page[k] for k in ("source", "page", "doc_id")},
                    "text":      chunk_text,
                    "chunk_id":  f"{page['doc_id']}_p{page['page']}_c{chunk_idx}",
                    "chunk_idx": chunk_idx,
                    "strategy":  "semantic",
                })

    mode = "semantic+similarity" if use_similarity else "semantic(no-sim)"
    print(f"Semantic chunker [{mode}]: {len(pages)} pages → {len(chunks)} chunks")
    return chunks


def chunk_documents(pages: List[Dict[str, Any]],
                    strategy: str = "semantic") -> List[Dict[str, Any]]:
    """Entry point — choose chunking strategy."""
    if strategy == "fixed":
        return chunk_fixed(pages)
    elif strategy == "semantic":
        return chunk_semantic(pages)
    else:
        raise ValueError(f"Unknown strategy: {strategy}. Use 'fixed' or 'semantic'.")
