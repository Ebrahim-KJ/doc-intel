"""
llm_client.py
-------------
Anthropic Claude API client.
Supports both streaming (for the dashboard) and batch (for evaluation).
"""

import os
from typing import List, Dict, Iterator
import anthropic

CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
MAX_TOKENS   = 1500

_client = anthropic.Anthropic()


def complete(
    messages:    List[Dict],
    system:      str = "",
    max_tokens:  int = MAX_TOKENS,
    temperature: float = 0.1,
) -> str:
    """Non-streaming completion. Returns full response text."""
    response = _client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    )
    return response.content[0].text


def stream(
    messages:    List[Dict],
    system:      str = "",
    max_tokens:  int = MAX_TOKENS,
    temperature: float = 0.1,
) -> Iterator[str]:
    """
    Streaming completion. Yields text chunks as they arrive.
    Use in Streamlit with st.write_stream().
    """
    with _client.messages.stream(
        model=CLAUDE_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system,
        messages=messages,
    ) as s:
        for text in s.text_stream:
            yield text
