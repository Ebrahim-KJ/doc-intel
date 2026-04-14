"""
conversation.py
---------------
Sliding window conversation buffer.
Keeps the last N turns in memory to give the LLM context
about prior exchanges in the session.
"""

import os
from typing import List, Dict
from collections import deque

WINDOW_SIZE = int(os.getenv("CONVERSATION_WINDOW", 6))   # turns (user + assistant)


class ConversationMemory:
    """
    Fixed-size deque of message dicts:
      { "role": "user"|"assistant", "content": str }
    """

    def __init__(self, window: int = WINDOW_SIZE):
        self.window   = window
        self._history = deque(maxlen=window * 2)  # *2: each turn = 1 user + 1 assistant

    def add_user(self, text: str) -> None:
        self._history.append({"role": "user", "content": text})

    def add_assistant(self, text: str) -> None:
        self._history.append({"role": "assistant", "content": text})

    def get_messages(self) -> List[Dict]:
        """Return history as Anthropic-compatible message list."""
        return list(self._history)

    def clear(self) -> None:
        self._history.clear()

    def __len__(self) -> int:
        return len(self._history)


# Session-keyed memory store
_sessions: Dict[str, ConversationMemory] = {}


def get_memory(session_id: str) -> ConversationMemory:
    if session_id not in _sessions:
        _sessions[session_id] = ConversationMemory()
    return _sessions[session_id]


def clear_memory(session_id: str) -> None:
    if session_id in _sessions:
        _sessions[session_id].clear()
