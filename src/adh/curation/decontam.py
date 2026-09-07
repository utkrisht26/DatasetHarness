"""Train/test leakage guard (OUTLINE §2.4, PHASE0-CONTRACT §0.3 invariant).

Builds an n-gram index from the held-out eval sets named in ``decontaminate_against``
and flags any candidate that shares an n-gram with them (``contamination_flag``).
Decontamination is **mandatory before anything enters an eval split**; the format
stage uses the flag to keep contaminated records out of validation/test.

Two detection layers:

* **Exact n-gram** (deterministic core) — any shared ``ngram``-gram with a held-out
  span flags the candidate outright.
* **Paraphrase gate** (OUTLINE §2.4 step 9, deterministic stand-in for LLM-paraphrase
  decontamination) — normalized **token-set Jaccard** plus **contiguous
  ``paraphrase_ngram``-gram containment ratio** against each held-out span. Catches
  rephrased copies (synonym swaps, light rewording) that defeat every exact window:
  a single well-placed synonym can break all shared 13-grams while leaving the token
  sets near-identical. Below :data:`MIN_FUZZY_TOKENS` there is too little signal for
  reliable fuzzy matching, so only the exact index decides.

Thresholds come from ``spec.dedup_policy`` when built via :meth:`from_spec`
(``jaccard`` and ``ngram`` double as the paraphrase-gate knobs), else the documented
:data:`DEFAULT_PARAPHRASE_*` defaults apply.
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

# Sane paraphrase-gate defaults when no spec config supplies them. 0.7 token-set
# Jaccard / 0.5 contiguous 8-gram containment flag rephrased copies while sparing
# topically-similar-but-distinct text; below 6 tokens fuzzy matching is noise.
DEFAULT_PARAPHRASE_JACCARD = 0.7
DEFAULT_PARAPHRASE_OVERLAP = 0.5
DEFAULT_PARAPHRASE_NGRAM = 8
MIN_FUZZY_TOKENS = 6


def _ngrams(text: str, n: int) -> set[str]:
    tokens = _WORD_RE.findall((text or "").lower())
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


@dataclass
class Decontaminator:
    """Flags candidates overlapping the held-out eval set (exact + paraphrase)."""

    ngram: int = 13
    paraphrase_jaccard: float = DEFAULT_PARAPHRASE_JACCARD
    paraphrase_overlap: float = DEFAULT_PARAPHRASE_OVERLAP
    paraphrase_ngram: int = DEFAULT_PARAPHRASE_NGRAM
    min_fuzzy_tokens: int = MIN_FUZZY_TOKENS
    _index: set[str] = field(default_factory=set)
    # per held-out span: (normalized token set, contiguous n-gram set)
    _ref_spans: list[tuple[frozenset[str], frozenset[str]]] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return bool(self._index or self._ref_spans)

    @classmethod
    def from_paths(
        cls,
        paths: list[str],
        *,
        ngram: int = 13,
        base_dir: Path | None = None,
        paraphrase_jaccard: float = DEFAULT_PARAPHRASE_JACCARD,
        paraphrase_overlap: float = DEFAULT_PARAPHRASE_OVERLAP,
        paraphrase_ngram: int = DEFAULT_PARAPHRASE_NGRAM,
    ) -> "Decontaminator":
        d = cls(
            ngram=ngram,
            paraphrase_jaccard=paraphrase_jaccard,
            paraphrase_overlap=paraphrase_overlap,
            paraphrase_ngram=paraphrase_ngram,
        )
        for ref in paths:
            p = Path(ref)
            if not p.is_absolute() and base_dir is not None:
                p = base_dir / p
            if not p.exists():
                logger.warning("decontaminate_against path missing: %s", ref)
                continue
            for text in _iter_texts(p):
                d.add(text)
        return d

    @classmethod
    def from_spec(cls, spec, *, base_dir: Path | None = None) -> "Decontaminator":
        """Build from a JobSpec; thresholds come from ``spec.dedup_policy``.

        ``dedup_policy.jaccard`` becomes the token-set Jaccard threshold and
        ``dedup_policy.ngram`` the containment gram size; fields absent on the
        policy (or no policy at all) fall back to the documented module defaults.
        The exact-match index keeps its own conservative default (13-grams).
        """
        policy = getattr(spec, "dedup_policy", None)
        return cls.from_paths(
            list(getattr(spec, "decontaminate_against", None) or []),
            ngram=13,
            base_dir=base_dir,
            paraphrase_jaccard=(
                float(getattr(policy, "jaccard", DEFAULT_PARAPHRASE_JACCARD))
                if policy is not None
                else DEFAULT_PARAPHRASE_JACCARD
            ),
            paraphrase_ngram=(
                int(getattr(policy, "ngram", DEFAULT_PARAPHRASE_NGRAM))
                if policy is not None
                else DEFAULT_PARAPHRASE_NGRAM
            ),
        )

    def add(self, text: str) -> None:
        """Ingest one held-out span into both the exact index and the fuzzy refs."""
        tokens = _WORD_RE.findall((text or "").lower())
        if not tokens:
            return
        self._index |= _ngrams(text, self.ngram)
        self._ref_spans.append((frozenset(tokens), _ngrams(text, self.paraphrase_ngram)))

    def contamination_reason(self, text: str) -> str | None:
        """Why the text is contaminated, or ``None`` — exact hit first, then gates."""
        if not self.active:
            return None
        if _ngrams(text, self.ngram) & self._index:
            return f"exact {self.ngram}-gram overlap with held-out set"
        tokens = _WORD_RE.findall((text or "").lower())
        if len(tokens) < self.min_fuzzy_tokens:
            return None
        token_set = frozenset(tokens)
        cand_grams = _ngrams(text, self.paraphrase_ngram)
        for pos, (ref_tokens, ref_grams) in enumerate(self._ref_spans):
            union = len(token_set | ref_tokens)
            jaccard = len(token_set & ref_tokens) / union if union else 0.0
            if jaccard >= self.paraphrase_jaccard:
                return (
                    f"paraphrase of held-out span {pos}: token-set Jaccard "
                    f"{jaccard:.2f} >= threshold {self.paraphrase_jaccard:.2f}"
                )
            if cand_grams and ref_grams:
                containment = len(cand_grams & ref_grams) / len(cand_grams)
                if containment >= self.paraphrase_overlap:
                    return (
                        f"paraphrase of held-out span {pos}: {self.paraphrase_ngram}-gram "
                        f"containment {containment:.2f} >= threshold {self.paraphrase_overlap:.2f}"
                    )
        return None

    def is_contaminated(self, text: str) -> bool:
        return self.contamination_reason(text) is not None


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
