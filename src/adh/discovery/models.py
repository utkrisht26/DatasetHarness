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

    def to_dict(self) -> dict:
        return {
            "aspect_id": self.aspect_id,
            "topic": self.topic,
            "rationale": self.rationale,
            "keywords": list(self.keywords),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Aspect":
        return cls(
            aspect_id=data["aspect_id"],
            topic=data["topic"],
            rationale=data.get("rationale", ""),
            keywords=tuple(data.get("keywords", [])),
        )


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

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "channel": self.channel.value,
            "locator": self.locator,
            "title": self.title,
            "aspect_id": self.aspect_id,
            "snippet": self.snippet,
            "license": self.license,
            "published": self.published,
            "discovered_via": self.discovered_via,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CandidateSource":
        return cls(
            source_id=data["source_id"],
            channel=SourceChannel(data["channel"]),
            locator=data["locator"],
            title=data.get("title", ""),
            aspect_id=data.get("aspect_id", ""),
            snippet=data.get("snippet", ""),
            license=data.get("license"),
            published=data.get("published"),
            discovered_via=data.get("discovered_via", ""),
        )


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

    def to_dict(self) -> dict:
        return {
            "source_id": self.source_id,
            "shape_match": self.shape_match,
            "quality": self.quality,
            "license_ok": self.license_ok,
            "freshness_ok": self.freshness_ok,
            "estimated_yield": self.estimated_yield,
            "score": self.score,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SourceEvaluation":
        return cls(
            source_id=data["source_id"],
            shape_match=float(data["shape_match"]),
            quality=float(data["quality"]),
            license_ok=bool(data["license_ok"]),
            freshness_ok=bool(data["freshness_ok"]),
            estimated_yield=int(data["estimated_yield"]),
            score=float(data["score"]),
            notes=data.get("notes", ""),
        )


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

    def to_dict(self) -> dict:
        return {"candidate": self.candidate.to_dict(), "evaluation": self.evaluation.to_dict()}

    @classmethod
    def from_dict(cls, data: dict) -> "RankedSource":
        return cls(
            candidate=CandidateSource.from_dict(data["candidate"]),
            evaluation=SourceEvaluation.from_dict(data["evaluation"]),
        )


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

    def to_dict(self) -> dict:
        """JSON form for ``plan.json`` — the plan must survive the approval pause."""
        return {
            "job_id": self.job_id,
            "sources": [s.to_dict() for s in self.sources],
            "synthesis_aspects": [a.to_dict() for a in self.synthesis_aspects],
            "aspects": [a.to_dict() for a in self.aspects],
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "FetchPlan":
        return cls(
            job_id=data["job_id"],
            sources=tuple(RankedSource.from_dict(s) for s in data.get("sources", [])),
            synthesis_aspects=tuple(Aspect.from_dict(a) for a in data.get("synthesis_aspects", [])),
            aspects=tuple(Aspect.from_dict(a) for a in data.get("aspects", [])),
            notes=list(data.get("notes", [])),
        )
