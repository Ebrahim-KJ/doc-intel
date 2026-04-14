"""
prompt_templates.py
-------------------
System prompts and context assembly for the RAG pipeline.
"""

from typing import List, Dict, Any

SYSTEM_PROMPT = """You are DocIntel, an expert research assistant powered by Retrieval-Augmented Generation.

Your role is to answer questions accurately and concisely based ONLY on the provided document context.

Rules:
1. Base your answer strictly on the retrieved context below. Do not use outside knowledge.
2. Cite your sources inline using [Source: <filename>, Page <n>] immediately after each claim.
3. If the context does not contain enough information to answer the question, say so clearly — do not guess.
4. Be concise. Prefer bullet points for multi-part answers.
5. If you quote directly from the context, use quotation marks.

Retrieved context:
{context}"""


def format_context(chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("source", "unknown")
        page   = chunk.get("page", "?")
        text   = chunk["text"].strip()
        lines.append(f"[{i}] Source: {source}, Page {page}\n{text}")
    return "\n\n---\n\n".join(lines)


def build_system_prompt(chunks: List[Dict[str, Any]]) -> str:
    """Inject retrieved context into the system prompt."""
    context = format_context(chunks)
    return SYSTEM_PROMPT.format(context=context)


def extract_citations(chunks: List[Dict[str, Any]]) -> List[Dict]:
    """Return citation metadata for display in the UI."""
    seen = set()
    citations = []
    for chunk in chunks:
        key = (chunk.get("source", ""), chunk.get("page", 0))
        if key not in seen:
            seen.add(key)
            citations.append({
                "source":   chunk.get("source", "unknown"),
                "page":     chunk.get("page", "?"),
                "doc_id":   chunk.get("doc_id", ""),
                "preview":  chunk["text"][:120] + "...",
            })
    return citations
