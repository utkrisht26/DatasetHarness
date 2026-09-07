"""Coverage-gap loop driver (OUTLINE §1.0) — discovery is a loop, not a one-shot.

Composes both engines into the feedback cycle the OUTLINE names:

    build (Engine B) -> measure per-aspect coverage -> re-discover sources for
    thin aspects -> acquire *only* the new sources (Engine A resume skips every
    already-completed locator via durable state) -> rebuild -> repeat. The merged
    plan (with fresh re-discovered sources) is persisted *before* the approver
    runs — same as initial plan_run — so approval filters what a round acquires
    but never rewrites the audit trail.

Termination is guaranteed four ways: the gap closes (``coverage_closed``),
re-discovery stops yielding fresh locators (``no_new_sources``), the round
budget runs out (``max_rounds``), or an operator/circuit-breaker stop arrives
(``cancelled`` / ``reacquisition_stopped``). Every transition is persisted —
plan, acquisition state, verified-candidate journal — so killing the loop at
any point resumes without redoing paid work.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

from adh.channels.base import Channel
from adh.contract import JobSpec, SourceChannel
from adh.corpus import CorpusStore
from adh.discovery.approval import AutoApprover, Approver
from adh.discovery.models import Aspect, FetchPlan
from adh.generation.generator import Generator
from adh.llm import UsageTracker
from adh.verification import JudgePanel

from .engine_a import EngineAResult, resume_run
from .engine_b import EngineBResult, run_engine_b
from .run_store import load_plan, load_spec, save_plan, save_spec

logger = logging.getLogger(__name__)


@runtime_checkable
class Rediscoverer(Protocol):
    """The slice of the discovery agent the loop needs (test-seam friendly)."""

    def rediscover(self, spec: JobSpec, aspect: Aspect, *, existing: FetchPlan) -> FetchPlan:
        """Return ``existing`` merged with any fresh sources found for ``aspect``."""
        ...


@dataclass
class CoverageLoopResult:
    """Outcome of the build/re-discover/re-acquire cycle."""

    job_id: str
    rounds: int  # completed re-discovery + re-acquisition cycles
    exit_reason: str  # coverage_closed | no_new_sources | max_rounds | cancelled | reacquisition_stopped
    result: EngineBResult  # final Engine B build over the final corpus
    acquisitions: list[EngineAResult] = field(default_factory=list)
    thin_aspects: tuple[str, ...] = ()  # still-thin aspects when the loop stopped
    notes: list[str] = field(default_factory=list)


def run_coverage_loop(
    spec: JobSpec,
    *,
    work_dir: str | Path,
    agent: Rediscoverer,
    channels: dict[SourceChannel, Channel] | None = None,
    approver: Approver | None = None,
    generator: Generator | None = None,
    judge: JudgePanel | None = None,
    store: CorpusStore | None = None,
    usage: UsageTracker | None = None,
    max_rounds: int = 2,
    should_cancel: Callable[[], bool] | None = None,
    schema_base_dir: Path | None = None,
) -> CoverageLoopResult:
    """Drive build -> gap re-discovery -> incremental acquisition -> rebuild.

    Requires ``work_dir`` to hold a completed Engine A run (``plan.json`` +
    manifest); raises ``ValueError`` otherwise. Engine B always resumes its
    verified journal here — that continuity is the loop's whole point — so no
    ``resume`` flag is offered. The default :class:`AutoApprover` approves the
    full merged plan each round; durable state makes that incremental (done
    sources are never refetched), while a stricter ``Approver`` filters what
    this round acquires exactly as it would at initial approval.
    """
    work = Path(work_dir)
    job_id = spec.job_id or spec.name
    approver = approver or AutoApprover()
    notes: list[str] = []
    acquisitions: list[EngineAResult] = []

    plan = load_plan(work)
    if plan is None:
        raise ValueError(
            f"coverage loop needs an Engine A work directory containing plan.json "
            f"(got {work}); run plan_run/resume_run (or run_engine_a) first"
        )

    # resume_run re-loads the persisted spec snapshot, so a stale spec.json would
    # silently govern re-acquisition under old rules while discovery/build used
    # the caller's YAML. Re-save on divergence: one shared contract everywhere.
    if load_spec(work) != spec:
        save_spec(work, spec)
        notes.append(
            "work-dir spec.json differed from the given spec; re-saved so "
            "re-acquisition, build and re-discovery share one contract"
        )

    def _build() -> EngineBResult:
        return run_engine_b(
            spec,
            work_dir=work,
            generator=generator,
            judge=judge,
            store=store,
            usage=usage,
            schema_base_dir=schema_base_dir,
            resume=True,
        )

    def _outcome(exit_reason: str) -> CoverageLoopResult:
        return CoverageLoopResult(
            job_id=job_id,
            rounds=rounds,
            exit_reason=exit_reason,
            result=result,
            acquisitions=acquisitions,
            thin_aspects=tuple(result.coverage.thin_aspects),
            notes=notes,
        )

    result = _build()
    rounds = 0
    while True:
        if should_cancel is not None and should_cancel():
            return _outcome("cancelled")
        thin = tuple(result.coverage.thin_aspects)
        if not thin:
            return _outcome("coverage_closed")
        if rounds >= max_rounds:
            notes.append(
                f"round budget exhausted ({max_rounds}); still thin: {', '.join(thin)}"
            )
            return _outcome("max_rounds")

        merged, fresh_ids, dry_aspects = _rediscover_thin(spec, plan, thin, agent, notes)
        if dry_aspects:
            notes.append(f"re-discovery found nothing new for: {', '.join(dry_aspects)}")
        if not fresh_ids:
            notes.append("no new sources surfaced for any thin aspect; stopping")
            return _outcome("no_new_sources")

        save_plan(work, merged)  # audit trail + the copy resume_run loads
        decision = approver.review(merged)
        acquired = resume_run(
            work, decision, channels=channels, store=store,
            resume=True, should_cancel=should_cancel,
        )
        acquisitions.append(acquired)
        rounds += 1
        notes.append(
            f"round {rounds}: +{len(fresh_ids)} source(s) for {len(thin)} thin aspect(s); "
            f"acquisition={acquired.exit_reason}"
        )
        if acquired.exit_reason != "completed":
            # Fold whatever arrived into the dataset, then surface the stop loudly.
            result = _build()
            if acquired.exit_reason == "cancelled":
                return _outcome("cancelled")
            notes.append(f"re-acquisition stopped early ({acquired.exit_reason})")
            return _outcome("reacquisition_stopped")
        result = _build()
        plan = merged


def _rediscover_thin(
    spec: JobSpec,
    plan: FetchPlan,
    thin: tuple[str, ...],
    agent: Rediscoverer,
    notes: list[str],
) -> tuple[FetchPlan, list[str], list[str]]:
    """Ask the agent for fresh sources per thin aspect.

    Returns ``(merged_plan, fresh_source_ids, aspects_that_yielded_nothing)``.
    An aspect absent from the plan (synthesis-only taxonomy drift) is noted and
    skipped rather than silently dropped.
    """
    by_id = {a.aspect_id: a for a in plan.aspects}
    known_ids = {s.source_id for s in plan.sources}
    merged = plan
    fresh_ids: list[str] = []
    dry: list[str] = []
    for aid in thin:
        aspect = by_id.get(aid)
        if aspect is None:
            notes.append(f"thin aspect {aid!r} has no fetch-plan counterpart; skipping")
            continue
        before_locators = {s.candidate.locator for s in merged.sources}
        candidate_plan = agent.rediscover(spec, aspect, existing=merged)
        fresh = [
            s
            for s in candidate_plan.sources
            if s.candidate.locator not in before_locators and s.source_id not in known_ids
        ]
        if not fresh:
            dry.append(aid)
            continue
        fresh_ids.extend(s.source_id for s in fresh)
        known_ids.update(s.source_id for s in fresh)
        merged = candidate_plan
    return merged, fresh_ids, dry
