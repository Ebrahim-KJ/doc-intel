"""
multi_query.py
--------------
Uses the LLM to generate N alternative phrasings of the user's query,
then runs retrieval on all variants and deduplicates results.

This addresses the brittleness of single-query retrieval:
a user asking "what causes inflation?" might miss relevant chunks
that discuss "price level increases" or "monetary expansion".
"""

import os
from typing import List
import anthropic

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
N_VARIANTS   = 3

_client = anthropic.Anthropic()

EXPANSION_PROMPT = """You are a query expansion assistant for a document retrieval system.

Given a user query, generate {n} alternative phrasings that capture the same information need
but use different vocabulary, framing, or specificity levels.

User query: {query}

Respond with ONLY {n} alternative queries, one per line, no numbering or bullet points."""


def expand_query(query: str, n: int = N_VARIANTS) -> List[str]:
    """
    Generate n alternative query formulations using Claude.
    Returns original query + n variants (deduplicated).
    """
    try:
        message = _client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=300,
            messages=[{
                "role": "user",
                "content": EXPANSION_PROMPT.format(query=query, n=n)
            }]
        )
        text     = message.content[0].text.strip()
        variants = [line.strip() for line in text.split("\n") if line.strip()][:n]
    except Exception as e:
        print(f"⚠️  Query expansion failed ({e}), using original query only")
        variants = []

    all_queries = [query] + variants
    seen        = set()
    unique      = []
    for q in all_queries:
        if q.lower() not in seen:
            seen.add(q.lower())
            unique.append(q)

    print(f"  Query expansion: 1 → {len(unique)} queries")
    return unique
