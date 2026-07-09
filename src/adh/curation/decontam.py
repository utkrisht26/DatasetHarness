"""Train/test leakage guard (OUTLINE §2.4, PHASE0-CONTRACT §0.3 invariant).

Builds an n-gram index from the held-out eval sets named in ``decontaminate_against``
and flags any candidate that shares an n-gram with them (``contamination_flag``).
Decontamination is **mandatory before anything enters an eval split**; the format
stage uses the flag to keep contaminated records out of validation/test. n-gram
overlap is the deterministic core; LLM-paraphrase decontamination (the final
eval-split gate) is a documented later addition.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")
_TEXT_KEYS = ("input", "output", "text", "question", "answer", "prompt", "completion")


def _ngrams(text: str, n: int) -> set[str]:
    tokens = _WORD_RE.findall((text or "").lower())
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class Decontaminator:
    """Flags candidates overlapping the held-out eval n-gram index."""

    ngram: int = 13
    _index: set[str] = field(default_factory=set)

    @property
    def active(self) -> bool:
        return bool(self._index)

    @classmethod
    def from_paths(cls, paths: list[str], *, ngram: int = 13, base_dir: Path | None = None) -> "Decontaminator":
        d = cls(ngram=ngram)
        for ref in paths:
            p = Path(ref)
            if not p.is_absolute() and base_dir is not None:
                p = base_dir / p
            if not p.exists():
                logger.warning("decontaminate_against path missing: %s", ref)
                continue
            for text in _iter_texts(p):
                d._index |= _ngrams(text, ngram)
        return d

    def is_contaminated(self, text: str) -> bool:
        if not self._index:
            return False
        return bool(_ngrams(text, self.ngram) & self._index)


def _iter_texts(path: Path):
    """Yield text spans from a held-out set (JSONL of records, or plain text)."""
    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                yield line
                continue
            if isinstance(obj, dict):
                parts = [str(obj[k]) for k in _TEXT_KEYS if k in obj and obj[k]]
                yield " ".join(parts) if parts else json.dumps(obj)
            else:
                yield str(obj)
    else:
        yield path.read_text(encoding="utf-8")
