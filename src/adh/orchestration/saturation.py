"""Coverage-saturation tracker — the PRIMARY stop signal (OUTLINE §1.4).

Stop collecting a topic when new pages stop adding novel information, *not* when an
arbitrary page count is hit (GOAL.md "coverage-driven, not budget-driven"). This is a
statistical approximation of Crawl4AI's ``AdaptiveCrawler`` (coverage / consistency /
saturation, ``confidence_threshold≈0.7``); production can delegate to the real crawler.

Novelty per chunk = fraction of its word-shingles not seen before. Saturation
confidence = 1 − novelty. A topic is saturated once it has warmed up (``min_chunks``)
and the mean novelty over the recent window falls at/below ``1 − confidence_threshold``.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

_SHINGLE_N = 5


def _shingles(text: str, n: int = _SHINGLE_N) -> set[str]:
    words = text.lower().split()
    if len(words) < n:
        return {" ".join(words)} if words else set()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


@dataclass
class SaturationTracker:
    confidence_threshold: float = 0.7  # AdaptiveConfig default
    min_chunks: int = 3  # warm-up before saturation can trip
    window: int = 3  # recent chunks averaged

    _seen: set[str] = field(default_factory=set)
    _recent_novelty: deque[float] = field(default_factory=lambda: deque(maxlen=8))
    _chunks: int = 0

    def update(self, text: str) -> float:
        """Record a chunk; return its novelty in ``[0, 1]``."""
        incoming = _shingles(text)
        if not incoming:
            novelty = 0.0
        else:
            new = incoming - self._seen
            novelty = len(new) / len(incoming)
            self._seen |= incoming
        self._recent_novelty.append(novelty)
        self._chunks += 1
        return novelty

    @property
    def confidence(self) -> float:
        """Current saturation confidence = 1 − recent mean novelty."""
        if not self._recent_novelty:
            return 0.0
        recent = list(self._recent_novelty)[-self.window :]
        return 1.0 - (sum(recent) / len(recent))

    def is_saturated(self) -> bool:
        return self._chunks >= self.min_chunks and self.confidence >= self.confidence_threshold

    @property
    def chunks_seen(self) -> int:
        return self._chunks
