"""Reasoning port for the Source Discovery Agent.

Discovery is "the intelligent core" (OUTLINE §1.0) — a research loop over search and
registry APIs. The *loop* is the harness's contribution; the *reasoning steps* are a
port so they can be backed by a deepagents/LangGraph agent in production
(:class:`adh.llm.deepagents_provider`) or by a deterministic heuristic offline.

Keeping these four steps behind a Protocol is what lets the whole discovery pipeline
run and be tested without a live model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

# Tokens that carry no topical signal — dropped when deriving keywords/aspects.
_STOPWORDS = frozenset(
    """
    a an the of for to and or with without on in into from by as is are be teach model
    data dataset datasets records record pairs pair patterns design how do does idiomatic
    advanced this that these those it its their our your build me want need about over
    """.split()
)
_SPLIT_RE = re.compile(r"[.;:\n]|(?:\b(?:and|plus|including|as well as|&)\b)", re.IGNORECASE)
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_+\-/.]*")


@dataclass(frozen=True)
class AspectDraft:
    """A sub-topic of the goal proposed by the reasoning step (pre-id-assignment)."""

    topic: str
    rationale: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class CandidateAssessment:
    """How well a discovered candidate matches the target record shape + quality bar."""

    shape_match: float  # 0..1 — does it actually contain records of the target shape?
    quality: float  # 0..1 — signal/quality estimate from the sampled content
    est_yield: int  # estimated usable records this source can produce
    notes: str = ""


@runtime_checkable
class LLMProvider(Protocol):
    """The reasoning steps the discovery loop delegates."""

    def decompose(self, goal: str, *, max_aspects: int = 6) -> list[AspectDraft]:
        """Break a plain-English goal into distinct aspects / sub-topics."""
        ...

    def queries(self, aspect: AspectDraft, *, n: int = 4) -> list[str]:
        """Generate diverse search queries for an aspect."""
        ...

    def assess_candidate(
        self, *, target_shape: str, keywords: tuple[str, ...], sample_text: str
    ) -> CandidateAssessment:
        """Judge a sampled candidate source against the target shape."""
        ...

    def domain_is_model_known(self, aspect: AspectDraft) -> bool:
        """Whether the model can synthesize this aspect with no external source."""
        ...


def _keywords(text: str, *, limit: int = 8) -> tuple[str, ...]:
    seen: list[str] = []
    for raw in _WORD_RE.findall(text.lower()):
        if raw in _STOPWORDS or len(raw) < 2:
            continue
        if raw not in seen:
            seen.append(raw)
        if len(seen) >= limit:
            break
    return tuple(seen)


@dataclass
class HeuristicLLMProvider:
    """Deterministic, network-free :class:`LLMProvider` for offline runs and tests.

    It is intentionally simple but *real*: it decomposes on conjunction boundaries,
    builds keyword-overlap queries, and scores candidates by lexical overlap between
    the aspect keywords and the sampled content. Production swaps in a model-backed
    provider; this keeps the orchestration honest and reproducible.
    """

    # Domains the heuristic treats as "model-known" (synthesis-viable) by keyword.
    synthesis_known_terms: frozenset[str] = field(
        default_factory=lambda: frozenset(
            {"rust", "python", "sql", "math", "algebra", "macro", "async", "code"}
        )
    )

    def decompose(self, goal: str, *, max_aspects: int = 6) -> list[AspectDraft]:
        chunks = [c.strip() for c in _SPLIT_RE.split(goal) if c and c.strip()]
        # Deduplicate near-identical chunks while preserving order.
        drafts: list[AspectDraft] = []
        seen_topics: set[str] = set()
        for chunk in chunks:
            kws = _keywords(chunk)
            if not kws:
                continue
            topic = " ".join(kws[:4])
            if topic in seen_topics:
                continue
            seen_topics.add(topic)
            drafts.append(
                AspectDraft(
                    topic=topic,
                    rationale=f"Derived from goal fragment: {chunk[:80]!r}",
                    keywords=kws,
                )
            )
            if len(drafts) >= max_aspects:
                break
        if not drafts:  # fall back to a single whole-goal aspect
            kws = _keywords(goal) or ("general",)
            drafts.append(
                AspectDraft(topic=" ".join(kws[:4]), rationale="Whole-goal aspect", keywords=kws)
            )
        return drafts

    def queries(self, aspect: AspectDraft, *, n: int = 4) -> list[str]:
        kws = list(aspect.keywords)
        out: list[str] = []
        if kws:
            out.append(" ".join(kws[:4]))
            out.append(" ".join(kws[:3]) + " examples")
            out.append(" ".join(kws[:3]) + " tutorial")
            out.append(" ".join(kws[:2]) + " dataset")
        seen: list[str] = []
        for q in out:
            if q and q not in seen:
                seen.append(q)
        return seen[:n]

    def assess_candidate(
        self, *, target_shape: str, keywords: tuple[str, ...], sample_text: str
    ) -> CandidateAssessment:
        sample_tokens = set(_WORD_RE.findall(sample_text.lower()))
        kw = set(keywords)
        overlap = len(kw & sample_tokens) / len(kw) if kw else 0.0
        # quality proxy: prose length + presence of Q/A-ish or code structure
        length_score = min(len(sample_text) / 2000.0, 1.0)
        shape_terms = set(_WORD_RE.findall(target_shape.lower()))
        shape_hit = len(shape_terms & sample_tokens) / len(shape_terms) if shape_terms else 0.0
        shape_match = round(0.6 * overlap + 0.4 * shape_hit, 4)
        quality = round(0.7 * length_score + 0.3 * overlap, 4)
        est_yield = int(round(shape_match * max(len(sample_text) // 200, 1)))
        return CandidateAssessment(
            shape_match=shape_match,
            quality=quality,
            est_yield=est_yield,
            notes=f"overlap={overlap:.2f} shape_hit={shape_hit:.2f}",
        )

    def domain_is_model_known(self, aspect: AspectDraft) -> bool:
        return bool(set(aspect.keywords) & self.synthesis_known_terms)
