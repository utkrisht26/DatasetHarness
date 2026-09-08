"""Heuristic quality filters — the cheapest verification stage (OUTLINE §2.3 step 2).

A deterministic Gopher/C4-style gate that rejects degenerate content before any
model or verifier is spent on it: too short, symbol soup, heavy line/n-gram
repetition, and (for prose) unnaturally low stopword ratio. Prose-specific checks are
relaxed for ``code`` modality. Thresholds are lenient by design — this stage removes
junk, it does not judge quality (that is the jury's job). RESEARCH §QC flags dialect
skew in stopword filters, so the stopword floor only applies to long-enough English.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

_WORD_RE = re.compile(r"\S+")
_STOPWORDS = frozenset(
    "the a an and or but of to in on for with is are was were be been it this that "
    "as at by from you i we they he she".split()
)


@dataclass(frozen=True)
class QualityResult:
    ok: bool
    reason: str = ""


@dataclass(frozen=True)
class QualityFilter:
    """Bounded, deterministic degenerate-content rejector."""

    min_words: int = 2
    max_words: int = 100_000
    max_symbol_ratio: float = 0.30
    max_dup_line_ratio: float = 0.35
    max_top_ngram_ratio: float = 0.25
    min_stopword_ratio: float = 0.03
    ngram: int = 8

    def check(self, text: str, *, modality: str = "text") -> QualityResult:
        text = (text or "").strip()
        if not text:
            return QualityResult(False, "empty output")
        words = _WORD_RE.findall(text)
        n = len(words)
        if n < self.min_words:
            return QualityResult(False, f"too short: {n} words < {self.min_words}")
        if n > self.max_words:
            return QualityResult(False, f"too long: {n} words > {self.max_words}")

        alpha = sum(c.isalnum() or c.isspace() for c in text)
        symbol_ratio = 1.0 - alpha / len(text)
        if symbol_ratio > self.max_symbol_ratio:
            return QualityResult(
                False, f"symbol ratio {symbol_ratio:.2f} > {self.max_symbol_ratio}"
            )

        dup = self._dup_line_ratio(text)
        if dup > self.max_dup_line_ratio:
            return QualityResult(
                False, f"duplicate-line ratio {dup:.2f} > {self.max_dup_line_ratio}"
            )

        top = self._top_ngram_ratio(words)
        if top > self.max_top_ngram_ratio:
            return QualityResult(False, f"repetition: top {self.ngram}-gram ratio {top:.2f}")

        # prose-only: natural English has stopwords; code does not.
        if modality != "code" and n >= 20:
            sw = sum(1 for w in words if w.lower().strip(".,;:!?") in _STOPWORDS) / n
            if sw < self.min_stopword_ratio:
                return QualityResult(False, f"stopword ratio {sw:.3f} < {self.min_stopword_ratio}")

        return QualityResult(True, "")

    def _dup_line_ratio(self, text: str) -> float:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) < 2:
            return 0.0
        counts = Counter(lines)
        duplicated = sum(c for c in counts.values() if c > 1)
        return duplicated / len(lines)

    def _top_ngram_ratio(self, words: list[str]) -> float:
        if len(words) < self.ngram * 2:
            return 0.0
        grams = Counter(
            tuple(words[i : i + self.ngram]) for i in range(len(words) - self.ngram + 1)
        )
        top = grams.most_common(1)[0][1]
        return top / sum(grams.values())
