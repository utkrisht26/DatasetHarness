"""The Source Discovery Agent — the intelligent core (OUTLINE §1.0).

Given only a goal + record shape (NO sources), answer "where does this data live?"
as a research loop over the search and registry ports:

    1. Decompose the goal into aspects.
    2. Route each aspect by channel: datasets (reuse) > web (crawl) > synthesis.
    3. Generate diverse queries, run search + registry lookups, collect candidates.
    4. Evaluate each candidate against the spec (shape / quality / license / freshness / yield).
    5. Rank + dedupe -> a FetchPlan.

Modes: ``auto`` (goal only), ``seeded`` (expand from hints), ``fixed`` (exact sources).
The approval gate and coverage-gap re-discovery live in sibling modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable
from urllib.parse import urlparse

from adh.contract import DiscoveryMode, JobSpec, SourceChannel
from adh.llm import AspectDraft, HeuristicLLMProvider, LLMProvider
from adh.search import (
    DatasetHit,
    DatasetRegistry,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
    SearchHit,
    SearchProvider,
)

from .models import Aspect, CandidateSource, FetchPlan, RankedSource, SourceEvaluation


@runtime_checkable
class DiscoveryAgent(Protocol):
    """The discovery seam: anything that turns a JobSpec into a FetchPlan.

    Two implementations (step-2 plan, decision 3): the hand-coded
    :class:`SourceDiscoveryAgent` (heuristic, offline/no-key fallback) and the
    deepagents-driven :class:`adh.discovery.deep_agent.DeepAgentDiscoveryAgent`.
    """

    def discover(self, spec: JobSpec, work_dir: Path) -> FetchPlan:
        """Produce a ranked, deduped fetch plan for ``spec``."""
        ...


@dataclass(frozen=True)
class DiscoveryConfig:
    """Tunable thresholds for routing + ranking (safe, documented defaults)."""

    queries_per_aspect: int = 4
    max_candidates_per_aspect: int = 8
    dataset_shape_threshold: float = 0.35  # reuse a dataset only if it plausibly matches
    web_shape_threshold: float = 0.15  # crawl when no dataset, lower bar
    # ranking weights
    w_shape: float = 0.5
    w_quality: float = 0.3
    w_yield: float = 0.2
    channel_bonus_datasets: float = 0.15  # reuse-first tie-break
    license_penalty: float = 0.5  # applied when license not in allowed_licenses
    freshness_penalty: float = 0.2


@dataclass
class SourceDiscoveryAgent:
    """Runs the discovery loop against injectable ports.

    Defaults are the offline/deterministic implementations so the agent is usable
    and testable with no configuration; production injects model- and network-backed
    providers (deepagents, HF Hub, SearXNG/Crawl4AI).
    """

    llm: LLMProvider = field(default_factory=HeuristicLLMProvider)
    search: SearchProvider = field(default_factory=InMemorySearchProvider)
    registry: DatasetRegistry = field(default_factory=InMemoryDatasetRegistry)
    config: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    # ------------------------------------------------------------------ public
    def discover(self, spec: JobSpec, work_dir: Path) -> FetchPlan:
        """Produce a ranked, deduped fetch plan for ``spec``."""
        job_id = spec.job_id or spec.name
        notes: list[str] = []

        if spec.discovery.mode == DiscoveryMode.FIXED:
            return self._fixed_plan(spec, job_id, notes)

        aspects = self._aspects(spec)
        ranked: list[RankedSource] = []
        synthesis_aspects: list[Aspect] = []

        for aspect in aspects:
            channel, sources = self._route_aspect(aspect, spec)
            if channel == SourceChannel.SYNTHESIS:
                synthesis_aspects.append(aspect)
                notes.append(f"aspect {aspect.aspect_id!r} routed to synthesis (model-known)")
            elif sources:
                ranked.extend(sources)
            else:
                notes.append(f"aspect {aspect.aspect_id!r} found no viable source (coverage gap)")

        ranked = self._dedupe_and_rank(ranked)
        return FetchPlan(
            job_id=job_id,
            sources=tuple(ranked),
            synthesis_aspects=tuple(synthesis_aspects),
            aspects=tuple(aspects),
            notes=notes,
        )

    def rediscover(self, spec: JobSpec, aspect: Aspect, *, existing: FetchPlan) -> FetchPlan:
        """Coverage-gap re-discovery: find more sources for one aspect and merge them.

        Engine B calls this when an aspect reports thin coverage (OUTLINE §1.0).
        Discovery is a loop, not a one-shot.
        """
        job_id = spec.job_id or spec.name
        seen = {s.candidate.locator for s in existing.sources}
        _, fresh = self._route_aspect(aspect, spec, exclude_locators=seen)
        merged = list(existing.sources) + [s for s in fresh if s.candidate.locator not in seen]
        merged = self._dedupe_and_rank(merged)
        notes = list(existing.notes) + [
            f"re-discovery for aspect {aspect.aspect_id!r}: +{len(fresh)} candidate(s)"
        ]
        return FetchPlan(
            job_id=job_id,
            sources=tuple(merged),
            synthesis_aspects=existing.synthesis_aspects,
            aspects=existing.aspects,
            notes=notes,
        )

    # ------------------------------------------------------------------ aspects
    def _aspects(self, spec: JobSpec) -> list[Aspect]:
        drafts = self.llm.decompose(self._goal_text(spec))
        return [
            Aspect(
                aspect_id=f"a{i}",
                topic=d.topic,
                rationale=d.rationale,
                keywords=d.keywords,
            )
            for i, d in enumerate(drafts)
        ]

    @staticmethod
    def _goal_text(spec: JobSpec) -> str:
        return goal_text(spec)

    @staticmethod
    def _target_shape(spec: JobSpec) -> str:
        return target_shape(spec)

    # ------------------------------------------------------------------ routing
    def _route_aspect(
        self,
        aspect: Aspect,
        spec: JobSpec,
        *,
        exclude_locators: set[str] | None = None,
    ) -> tuple[SourceChannel, list[RankedSource]]:
        """Route one aspect to a channel and return its evaluated sources.

        Reuse > crawl > synthesize, each gated by a shape-match threshold.
        """
        channels = spec.discovery.channels
        exclude = exclude_locators or set()
        draft = AspectDraft(aspect.topic, aspect.rationale, aspect.keywords)

        # 1) datasets channel (reuse first)
        if SourceChannel.DATASETS in channels:
            ds = self._datasets_for(aspect, draft, spec, exclude)
            passing = [s for s in ds if s.evaluation.shape_match >= self.config.dataset_shape_threshold]
            if passing:
                return SourceChannel.DATASETS, passing

        # 2) web channel (crawl when no ready dataset)
        if SourceChannel.WEB in channels:
            web = self._web_for(aspect, draft, spec, exclude)
            passing = [s for s in web if s.evaluation.shape_match >= self.config.web_shape_threshold]
            if passing:
                return SourceChannel.WEB, passing

        # 3) synthesis (model already knows the domain)
        if SourceChannel.SYNTHESIS in channels and self.llm.domain_is_model_known(draft):
            return SourceChannel.SYNTHESIS, []

        return SourceChannel.WEB, []  # nothing viable

    def _datasets_for(
        self, aspect: Aspect, draft: AspectDraft, spec: JobSpec, exclude: set[str]
    ) -> list[RankedSource]:
        out: list[RankedSource] = []
        for query in self.llm.queries(draft, n=self.config.queries_per_aspect):
            for hit in self.registry.search(query, limit=self.config.max_candidates_per_aspect):
                if hit.locator in exclude:
                    continue
                cand = CandidateSource(
                    source_id=f"{aspect.aspect_id}:ds:{hit.dataset_id}",
                    channel=SourceChannel.DATASETS,
                    locator=hit.locator,
                    title=hit.title,
                    aspect_id=aspect.aspect_id,
                    snippet=hit.description,
                    license=hit.license,
                    published=hit.last_modified,
                    discovered_via=query,
                )
                sample = self.registry.sample(hit.dataset_id) or hit.description
                out.append(self._evaluate(cand, draft, spec, sample, hit=hit))
        return out

    def _web_for(
        self, aspect: Aspect, draft: AspectDraft, spec: JobSpec, exclude: set[str]
    ) -> list[RankedSource]:
        out: list[RankedSource] = []
        for query in self.llm.queries(draft, n=self.config.queries_per_aspect):
            for hit in self.search.search(query, limit=self.config.max_candidates_per_aspect):
                if hit.url in exclude or not self._domain_allowed(hit.url, spec):
                    continue
                cand = CandidateSource(
                    source_id=f"{aspect.aspect_id}:web:{hit.url}",
                    channel=SourceChannel.WEB,
                    locator=hit.url,
                    title=hit.title,
                    aspect_id=aspect.aspect_id,
                    snippet=hit.snippet,
                    license=hit.license_hint,
                    published=hit.published,
                    discovered_via=query,
                )
                out.append(self._evaluate(cand, draft, spec, hit.snippet))
        return out

    # ------------------------------------------------------------------ eval
    def _evaluate(
        self,
        cand: CandidateSource,
        draft: AspectDraft,
        spec: JobSpec,
        sample_text: str,
        *,
        hit: DatasetHit | SearchHit | None = None,
    ) -> RankedSource:
        assessment = self.llm.assess_candidate(
            target_shape=self._target_shape(spec),
            keywords=draft.keywords,
            sample_text=sample_text,
        )
        lic_ok = license_ok(cand.license, spec)
        fresh_ok = freshness_ok(cand.published, spec)
        score = score_candidate(
            shape_match=assessment.shape_match,
            quality=assessment.quality,
            est_yield=assessment.est_yield,
            channel=cand.channel,
            license_ok=lic_ok,
            freshness_ok=fresh_ok,
            config=self.config,
        )
        evaluation = SourceEvaluation(
            source_id=cand.source_id,
            shape_match=assessment.shape_match,
            quality=assessment.quality,
            license_ok=lic_ok,
            freshness_ok=fresh_ok,
            estimated_yield=assessment.est_yield,
            score=round(score, 4),
            notes=assessment.notes,
        )
        return RankedSource(candidate=cand, evaluation=evaluation)

    # ------------------------------------------------------------------ filters
    def _license_ok(self, license_: str | None, spec: JobSpec) -> bool:
        return license_ok(license_, spec)

    def _freshness_ok(self, published: str | None, spec: JobSpec) -> bool:
        return freshness_ok(published, spec)

    def _domain_allowed(self, url: str, spec: JobSpec) -> bool:
        return domain_allowed(url, spec)

    # ------------------------------------------------------------------ rank
    def _dedupe_and_rank(self, sources: list[RankedSource]) -> list[RankedSource]:
        return dedupe_and_rank(sources)

    # ------------------------------------------------------------------ fixed
    def _fixed_plan(self, spec: JobSpec, job_id: str, notes: list[str]) -> FetchPlan:
        return fixed_plan(spec, job_id, notes)


# ------------------------------------------------------------- shared verdicts
# Deterministic evaluation shared by BOTH discovery agents (heuristic and deep):
# the model may judge shape/quality/yield, but license, freshness, domain policy,
# scoring, and ranking always stay harness-side.
def goal_text(spec: JobSpec) -> str:
    """The plain-English goal: description + motivation (+ seed hints when seeded)."""
    parts = [spec.description, spec.motivation]
    if spec.discovery.mode == DiscoveryMode.SEEDED and spec.discovery.seeds:
        parts.append("Hints: " + ", ".join(spec.discovery.seeds))
    return ". ".join(p for p in parts if p)


def target_shape(spec: JobSpec) -> str:
    return f"{' '.join(spec.task_categories)} {spec.modality} input output"


def license_ok(license_: str | None, spec: JobSpec) -> bool:
    if not license_:
        return False  # unknown license -> quarantine candidate (§0.8.3)
    allowed = {lic.lower() for lic in spec.allowed_licenses}
    return license_.lower() in allowed


def freshness_ok(published: str | None, spec: JobSpec) -> bool:
    after = spec.discovery.constraints.recency_after
    if not after or not published:
        return True  # unknown date: don't penalize
    return str(published)[:10] >= str(after)[:10]


def domain_allowed(url: str, spec: JobSpec) -> bool:
    host = (urlparse(url).hostname or "").lower()
    c = spec.discovery.constraints
    if any(host == d.lower() or host.endswith("." + d.lower()) for d in c.exclude_domains):
        return False
    if c.include_domains:
        return any(
            host == d.lower() or host.endswith("." + d.lower()) for d in c.include_domains
        )
    return True


def score_candidate(
    *,
    shape_match: float,
    quality: float,
    est_yield: int,
    channel: SourceChannel,
    license_ok: bool,
    freshness_ok: bool,
    config: DiscoveryConfig,
) -> float:
    """Aggregate ranking score: weighted assessment + channel bonus - policy penalties."""
    yield_norm = min(est_yield / 50.0, 1.0)
    base = config.w_shape * shape_match + config.w_quality * quality + config.w_yield * yield_norm
    if channel == SourceChannel.DATASETS:
        base += config.channel_bonus_datasets
    if not license_ok:
        base -= config.license_penalty
    if not freshness_ok:
        base -= config.freshness_penalty
    return max(base, 0.0)


def dedupe_and_rank(sources: list[RankedSource]) -> list[RankedSource]:
    """Best evaluation per locator, sorted best-first."""
    best: dict[str, RankedSource] = {}
    for s in sources:
        key = s.candidate.locator
        if key not in best or s.evaluation.score > best[key].evaluation.score:
            best[key] = s
    return sorted(best.values(), key=lambda s: s.evaluation.score, reverse=True)


def fixed_plan(spec: JobSpec, job_id: str, notes: list[str]) -> FetchPlan:
    """Degenerate mode: the user gave exact sources — just wrap them as a plan."""
    ranked: list[RankedSource] = []
    for i, seed in enumerate(spec.discovery.seeds):
        channel = SourceChannel.DATASETS if seed.startswith("hf://") else SourceChannel.WEB
        cand = CandidateSource(
            source_id=f"fixed:{i}",
            channel=channel,
            locator=seed,
            title=seed,
            aspect_id="fixed",
            discovered_via="fixed-seed",
        )
        evaluation = SourceEvaluation(
            source_id=cand.source_id,
            shape_match=1.0,
            quality=1.0,
            license_ok=True,
            freshness_ok=True,
            estimated_yield=0,
            score=1.0,
            notes="user-supplied fixed source",
        )
        ranked.append(RankedSource(candidate=cand, evaluation=evaluation))
    notes.append(f"fixed mode: {len(ranked)} user-supplied source(s)")
    return FetchPlan(job_id=job_id, sources=tuple(ranked), notes=notes)
