"""Token-usage accounting for LLM calls.

Feeds OUTLINE §3.1's ``tokens_used`` metric and the ``max_llm_tokens`` circuit-breaker
floor: any module that calls a model records usage into a shared :class:`UsageTracker`,
which the application reads to report cost and to fold into
:class:`adh.orchestration.limits.RunMetrics` (``metrics.add_tokens(tracker.total.total_tokens)``).

Token counts come from LangChain's provider-normalised ``AIMessage.usage_metadata`` (with
a ``response_metadata`` fallback for older providers), so the tracker is model-agnostic.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenUsage:
    """Immutable input/output token counts for one or more LLM calls."""

    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.input_tokens + other.input_tokens,
            self.output_tokens + other.output_tokens,
        )

    def to_dict(self) -> dict:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }

    @classmethod
    def from_message(cls, message: Any) -> "TokenUsage":
        """Extract usage from a LangChain response message (or anything resembling one).

        Prefers the provider-normalised ``usage_metadata``; falls back to the raw
        ``response_metadata`` token counts. Returns a zero usage if none are present.
        """
        meta = getattr(message, "usage_metadata", None)
        if isinstance(meta, dict):
            return cls(int(meta.get("input_tokens") or 0), int(meta.get("output_tokens") or 0))
        raw = getattr(message, "response_metadata", None) or {}
        usage = raw.get("token_usage") or raw.get("usage") or {}
        if usage:
            return cls(
                int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
                int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
            )
        return cls()


class UsageTracker:
    """Thread-safe accumulator of token usage across LLM calls.

    Judges run concurrently in the router's worker threads, so recording is locked.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._total = TokenUsage()
        self._calls = 0

    def add(self, usage: TokenUsage) -> None:
        with self._lock:
            self._total = self._total + usage
            self._calls += 1

    def add_message(self, message: Any) -> TokenUsage:
        """Record the usage carried by ``message`` and return that per-call delta."""
        usage = TokenUsage.from_message(message)
        self.add(usage)
        return usage

    @property
    def total(self) -> TokenUsage:
        with self._lock:
            return self._total

    @property
    def calls(self) -> int:
        with self._lock:
            return self._calls

    def to_dict(self) -> dict:
        with self._lock:
            return {**self._total.to_dict(), "calls": self._calls}
