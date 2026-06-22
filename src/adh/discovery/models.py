"""Domain models for the Source Discovery Agent (OUTLINE §1.0).

The discovery pipeline turns a goal into a ranked, deduped :class:`FetchPlan`:

    goal -> [Aspect] -> queries -> [CandidateSource] -> [SourceEvaluation]
         -> [RankedSource] (+ synthesis aspects) -> FetchPlan
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adh.contract import SourceChannel


@dataclass(frozen=True)
class Aspect:
    """A sub-topic of the goal, with a stable id (feeds the Phase 2.1 taxonomy too)."""

    aspect_id: str
    topic: str
    rationale: str
    keywords: tuple[str, ...]


@dataclass(frozen=True)
class CandidateSource:
    """A discovered source not yet evaluated against the spec."""

    source_id: str
    channel: SourceChannel
    locator: str  # URL, HF dataset id, or "synthesis://<aspect>"
    title: str
    aspect_id: str
    snippet: str = ""
    license: str | None = None
    published: str | None = None
    discovered_via: str = ""  # the query / registry that surfaced it


@dataclass(frozen=True)
class SourceEvaluation:
    """The verdict on a candidate after sampling it against the target shape."""

    source_id: str
    shape_match: float  # 0..1
    quality: float  # 0..1
    license_ok: bool
    freshness_ok: bool
    estimated_yield: int
    score: float  # aggregate ranking score
    notes: str = ""


@dataclass(frozen=True)
class RankedSource:
    """A candidate plus its evaluation, ready for the fetch plan."""

    candidate: CandidateSource
    evaluation: SourceEvaluation

    @property
    def channel(self) -> SourceChannel:
        return self.candidate.channel

    @property
    def source_id(self) -> str:
        return self.candidate.source_id


@dataclass(frozen=True)
class FetchPlan:
    """The output of discovery: what to fetch, in what order, and why."""

    job_id: str
    sources: tuple[RankedSource, ...] = ()  # ranked best-first, deduped
    synthesis_aspects: tuple[Aspect, ...] = ()  # routed to synthesis (no external source)
    aspects: tuple[Aspect, ...] = ()  # all aspects the goal decomposed into
    notes: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not self.sources and not self.synthesis_aspects

    def by_channel(self, channel: SourceChannel) -> tuple[RankedSource, ...]:
        return tuple(s for s in self.sources if s.channel == channel)
