"""The Source Discovery Agent — the intelligent core (OUTLINE §1.0).

Given only a goal + record shape (NO sources), answer "where does this data live?"
as a research loop over the search and registry ports:

    1. Decompose the goal into aspects.
    2. Generate and persist the complete, channel-scoped search plan.
    3. Route each aspect and execute the planned queries: datasets > web > synthesis.
    4. Evaluate each candidate against the spec (shape / quality / license / freshness / yield).
    5. Rank + dedupe -> a FetchPlan.

Modes: ``auto`` (goal only), ``seeded`` (expand from hints), ``fixed`` (exact sources).
The approval gate and coverage-gap re-discovery live in sibling modules.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable
from urllib.parse import urlparse

from adh.contract import (
    DiscoveryMode,
    JobSpec,
    SourceChannel,
    is_media_modality,
    resolved_media_types,
)
from adh.contract.licenses import normalize_license
from adh.junk import is_junk_source
from adh.llm import AspectDraft, HeuristicLLMProvider, LLMProvider
from adh.search import (
    DatasetHit,
    DatasetRegistry,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
    SearchHit,
    SearchProvider,
)
from adh.telemetry import current_telemetry

from .models import Aspect, CandidateSource, FetchPlan, RankedSource, SourceEvaluation

if TYPE_CHECKING:
    from adh.channels.media import MediaProvider


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
    media_shape_threshold: float = 0.15  # stock clips: same low bar as crawling
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
    # MEDIA channel port (media provider). None means the host has no media
    # stack — media-modality jobs then degrade to a loud coverage gap, never to
    # a text-about-media crawl.
    media_provider: MediaProvider | None = None
    config: DiscoveryConfig = field(default_factory=DiscoveryConfig)

    # ------------------------------------------------------------------ public
    def discover(self, spec: JobSpec, work_dir: Path) -> FetchPlan:
        """Produce a ranked, deduped fetch plan for ``spec``."""
        job_id = spec.job_id or spec.name
        notes: list[str] = []

        if spec.discovery.mode == DiscoveryMode.FIXED:
            return self._fixed_plan(spec, job_id, notes)

        aspects = self._aspects(spec)
        query_plan = self._query_plan(aspects)
        self._persist_search_plan(spec, work_dir, aspects, query_plan)
        ranked: list[RankedSource] = []
        synthesis_aspects: list[Aspect] = []

        for aspect in aspects:
            channel, sources = self._route_aspect(
                aspect,
                spec,
                queries=query_plan[aspect.aspect_id],
                notes=notes,
            )
            if channel == SourceChannel.SYNTHESIS:
                synthesis_aspects.append(aspect)
                notes.append(f"aspect {aspect.aspect_id!r} routed to synthesis (model-known)")
            elif channel == SourceChannel.MEDIA and not sources:
                notes.append(
                    f"aspect {aspect.aspect_id!r} routed to MEDIA but produced no sources "
                    "(no media provider configured, or no clips passed evaluation) — coverage gap"
                )
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
        skipped: list[str] = []
        _, fresh = self._route_aspect(aspect, spec, exclude_locators=seen, notes=skipped)
        merged = list(existing.sources) + [s for s in fresh if s.candidate.locator not in seen]
        merged = self._dedupe_and_rank(merged)
        notes = (
            list(existing.notes)
            + [f"re-discovery for aspect {aspect.aspect_id!r}: +{len(fresh)} candidate(s)"]
            + [n for n in skipped if n not in existing.notes]
        )
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

    def _query_plan(self, aspects: list[Aspect]) -> dict[str, tuple[str, ...]]:
        """Generate every aspect query before any provider is called."""
        plan: dict[str, tuple[str, ...]] = {}
        for aspect in aspects:
            draft = AspectDraft(aspect.topic, aspect.rationale, aspect.keywords)
            queries = self.llm.queries(draft, n=self.config.queries_per_aspect)
            plan[aspect.aspect_id] = tuple(dict.fromkeys(q.strip() for q in queries if q.strip()))
        return plan

    @staticmethod
    def _persist_search_plan(
        spec: JobSpec,
        work_dir: Path,
        aspects: list[Aspect],
        query_plan: dict[str, tuple[str, ...]],
    ) -> None:
        """Persist the heuristic agent's plan when running inside an observable job."""
        telemetry = current_telemetry()
        if telemetry is None:
            return

        enabled = set(spec.discovery.channels)
        searchable = enabled & {
            SourceChannel.DATASETS,
            SourceChannel.WEB,
            SourceChannel.MEDIA,
        }
        all_queries = list(
            dict.fromkeys(query for queries in query_plan.values() for query in queries)
        )
        queries = {
            channel.value: all_queries if channel in searchable else []
            for channel in (SourceChannel.DATASETS, SourceChannel.WEB, SourceChannel.MEDIA)
        }
        payload = {
            "agent": "heuristic-agent",
            "aspects": [
                {
                    "aspect_id": aspect.aspect_id,
                    "topic": aspect.topic,
                    "queries": list(query_plan[aspect.aspect_id]),
                }
                for aspect in aspects
            ],
            "queries": queries,
            "rationale": "deterministic fallback plan generated before provider search",
            "allowed_channels": sorted(channel.value for channel in enabled),
        }
        research_dir = Path(work_dir) / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        primary_path = research_dir / "search_plan.json"
        path = research_dir / "search_plan_fallback.json" if primary_path.exists() else primary_path
        attributes = {
            "artifact": str(path.relative_to(work_dir)),
            "aspects": len(aspects),
            "queries": len(all_queries),
            "channels": sorted(channel.value for channel in enabled),
        }
        with telemetry.span("planning", "search_plan", attributes=attributes):
            with path.open("w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2)

    # ------------------------------------------------------------------ routing
    def _route_aspect(
        self,
        aspect: Aspect,
        spec: JobSpec,
        *,
        exclude_locators: set[str] | None = None,
        queries: tuple[str, ...] | None = None,
        notes: list[str] | None = None,
    ) -> tuple[SourceChannel, list[RankedSource]]:
        """Route one aspect to a channel and return its evaluated sources.

        Skipped candidates are explained in ``notes`` (the plan's notes) when given.

        Reuse > crawl > synthesize, each gated by a shape-match threshold.
        """
        channels = spec.discovery.channels
        exclude = exclude_locators or set()
        draft = AspectDraft(aspect.topic, aspect.rationale, aspect.keywords)
        planned_queries = (
            queries
            if queries is not None
            else tuple(self.llm.queries(draft, n=self.config.queries_per_aspect))
        )

        # 0) media channel (media modalities: video/image): real assets, before
        #    any text path. Without a configured provider this reports a loud gap
        #    rather than silently crawling text *about* the media (the b6b176e6
        #    failure mode).
        if is_media_modality(spec.modality) and SourceChannel.MEDIA in channels:
            if self.media_provider is not None:
                media = self._media_for(aspect, draft, spec, exclude, planned_queries)
                passing = [
                    s
                    for s in media
                    if s.evaluation.shape_match >= self.config.media_shape_threshold
                ]
                if passing:
                    return SourceChannel.MEDIA, self._rank_and_limit(passing)
            return SourceChannel.MEDIA, []

        # 1) datasets channel (reuse first)
        if SourceChannel.DATASETS in channels:
            ds = self._datasets_for(aspect, draft, spec, exclude, planned_queries)
            passing = [
                s for s in ds if s.evaluation.shape_match >= self.config.dataset_shape_threshold
            ]
            if passing:
                return SourceChannel.DATASETS, self._rank_and_limit(passing)

        # 2) web channel (crawl when no ready dataset)
        if SourceChannel.WEB in channels:
            web = self._web_for(
                aspect, draft, spec, exclude, planned_queries, notes if notes is not None else []
            )
            passing = [
                s for s in web if s.evaluation.shape_match >= self.config.web_shape_threshold
            ]
            if passing:
                return SourceChannel.WEB, self._rank_and_limit(passing)

        # 3) synthesis (model already knows the domain)
        if SourceChannel.SYNTHESIS in channels and self.llm.domain_is_model_known(draft):
            return SourceChannel.SYNTHESIS, []

        return SourceChannel.WEB, []  # nothing viable

    def _datasets_for(
        self,
        aspect: Aspect,
        draft: AspectDraft,
        spec: JobSpec,
        exclude: set[str],
        queries: tuple[str, ...],
    ) -> list[RankedSource]:
        out: list[RankedSource] = []
        for query in queries:
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
        self,
        aspect: Aspect,
        draft: AspectDraft,
        spec: JobSpec,
        exclude: set[str],
        queries: tuple[str, ...],
        notes: list[str],
    ) -> list[RankedSource]:
        out: list[RankedSource] = []
        for query in queries:
            for hit in self.search.search(query, limit=self.config.max_candidates_per_aspect):
                if hit.url in exclude or not self._domain_allowed(hit.url, spec):
                    continue
                junk = is_junk_source(
                    hit.url, title=hit.title, snippet=hit.snippet, spec=spec, aspect=aspect
                )
                if junk:
                    note = f"skip {hit.url}: junk source ({junk.reason})"
                    if note not in notes:  # the same hit recurs across an aspect's queries
                        notes.append(note)
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

    def _media_for(
        self,
        aspect: Aspect,
        draft: AspectDraft,
        spec: JobSpec,
        exclude: set[str],
        queries: tuple[str, ...],
    ) -> list[RankedSource]:
        """Media-asset candidates; locator = search query (the channel re-searches)."""
        assert self.media_provider is not None  # guarded by the routing branch
        wanted = resolved_media_types(spec.modality, constraints=spec.discovery.constraints)
        out: list[RankedSource] = []
        seen_ids: set[str] = set()
        for query in queries:
            for hit in self.media_provider.search(
                query,
                limit=self.config.max_candidates_per_aspect,
                media_types=tuple(sorted(wanted)),
            ):
                if hit.media_id in seen_ids or hit.page_url in exclude:
                    continue
                seen_ids.add(hit.media_id)
                snippet = f"{hit.title} {', '.join(hit.tags)}".strip()
                cand = CandidateSource(
                    source_id=f"{aspect.aspect_id}:media:{hit.media_id}",
                    channel=SourceChannel.MEDIA,
                    locator=query,
                    title=hit.title,
                    aspect_id=aspect.aspect_id,
                    snippet=snippet,
                    license=hit.license,
                    discovered_via=query,
                )
                out.append(self._evaluate(cand, draft, spec, snippet))
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

    def _rank_and_limit(self, sources: list[RankedSource]) -> list[RankedSource]:
        """Keep the best unique candidates within the per-aspect search budget."""
        limit = max(self.config.max_candidates_per_aspect, 0)
        return self._dedupe_and_rank(sources)[:limit]

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
        return False  # missing info is not a declared unknown — stay quarantined
    normalized = normalize_license(license_)
    if normalized == "unknown":
        # Two-layer policy: open-web sources with unknown licenses only enter
        # when the spec explicitly opted in (distribution consent stays separate).
        return bool(spec.discovery.allow_unknown_licenses)
    allowed = {normalize_license(lic).lower() for lic in spec.allowed_licenses}
    return normalized.lower() in allowed


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
        return any(host == d.lower() or host.endswith("." + d.lower()) for d in c.include_domains)
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
        if key not in best or _rank_key(s) > _rank_key(best[key]):
            best[key] = s
    return sorted(best.values(), key=_rank_key, reverse=True)


def _rank_key(s: RankedSource) -> tuple[float, float, float, int]:
    """Score first; ties (e.g. several clamped to 0.0 by penalties) by assessment."""
    e = s.evaluation
    return (e.score, e.shape_match, e.quality, e.estimated_yield)


def fixed_plan(spec: JobSpec, job_id: str, notes: list[str]) -> FetchPlan:
    """Degenerate mode: the user gave exact sources — just wrap them as a plan."""
    ranked: list[RankedSource] = []
    for i, seed in enumerate(spec.discovery.seeds):
        if seed.startswith("hf://"):
            channel = SourceChannel.DATASETS
        elif seed.startswith(("pexels://", "media://")):
            channel = SourceChannel.MEDIA
        else:
            channel = SourceChannel.WEB
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
