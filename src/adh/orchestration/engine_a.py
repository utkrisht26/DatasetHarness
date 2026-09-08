"""Engine A driver — discovery -> approval -> acquisition -> corpus + manifest.

Ties the Phase 1 pieces into one resumable, limit-bounded run:

    discover (1.0) -> approval gate (1.0) -> per-channel fetch (1.1/1.6/1.7)
        with coverage-saturation stop (1.4) and the circuit-breaker floor (1.4)
        -> offloaded corpus (1.3) + manifest + lineage (3.3) + durable state (1.5)

The run is split into two phases around the approval gate (step-2 plan, T1):

* :func:`plan_run` — discover, meter the LLM spend, persist ``plan.json``, and
  park the run at ``awaiting_approval``. Holds NO thread while parked; the plan
  survives a restart.
* :func:`resume_run` — apply an :class:`ApprovalDecision` to the persisted plan
  and run the acquisition loop through to artifacts.

:func:`run_engine_a` composes the two (plan -> approve -> resume) so the CLI and
tests keep their one-call contract. The functions are *mechanism*: the caller
chooses the approver (the CLI prompts a human when ``approval_gate`` is on; the
service parks the run and waits for ``POST .../approve``; tests inject a policy).
Synthesis-routed aspects bypass acquisition and are recorded as hand-offs to
Engine B.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from adh.channels import FetchContext, handoffs_for
from adh.channels.base import Channel
from adh.contract import JobSpec, SourceChannel
from adh.corpus import CorpusChunk, CorpusStore, LocalCorpusStore, SynthesisHandoff
from adh.discovery import (
    ApprovalDecision,
    Approver,
    AutoApprover,
    DiscoveryAgent,
    FetchPlan,
    SourceDiscoveryAgent,
    apply_decision,
)
from adh.llm import UsageTracker
from adh.provenance import build_lineage, write_lineage

from .limits import CircuitBreaker, RunMetrics
from .run_store import (
    RUN_FILE,
    SUCCESS_EXIT_REASONS,
    RunRecord,
    RunStatus,
    fresh_run_id,
    load_plan,
    load_record,
    load_spec,
    new_record,
    save_plan,
    save_record,
    save_spec,
)
from .saturation import SaturationTracker
from .state import JobState, load_state, save_state

# Frontier channels the saturation governor may cut short (1.4): web crawls and
# media downloads both stream open-ended content for an aspect.
_SATURATION_CHANNELS = frozenset({SourceChannel.WEB, SourceChannel.MEDIA})


@dataclass
class EngineAResult:
    job_id: str
    plan: FetchPlan
    approved_plan: FetchPlan
    chunks: list[CorpusChunk]
    handoffs: list[SynthesisHandoff]
    metrics: RunMetrics
    exit_reason: str
    work_dir: Path
    manifest_path: Path
    lineage_path: Path
    summary_path: Path
    notes: list[str] = field(default_factory=list)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    @property
    def n_handoffs(self) -> int:
        return len(self.handoffs)


@dataclass
class PlannedRun:
    """Outcome of the discovery phase: the plan, parked at the approval gate."""

    record: RunRecord
    plan: FetchPlan
    work_dir: Path

    @property
    def awaiting_approval(self) -> bool:
        return self.record.status is RunStatus.AWAITING_APPROVAL


# --------------------------------------------------------------------- phase 1
def plan_run(
    spec: JobSpec,
    *,
    work_dir: str | Path,
    agent: DiscoveryAgent | None = None,
    usage: UsageTracker | None = None,
    resume: bool = True,
    run_id: str | None = None,
) -> PlannedRun:
    """Discover sources for ``spec`` and park the run at the approval gate.

    Persists ``spec.json`` + ``plan.json`` + the run record into ``work_dir`` so the
    pause survives a restart. Discovery LLM spend recorded in ``usage`` is folded
    into the durable :class:`RunMetrics`; if that already trips the circuit
    breaker (``max_llm_tokens``/``max_cost_usd``), a usable plan is still parked
    at ``awaiting_approval`` with the breaker noted — discovery was paid for, so
    a human decides whether to proceed. Only an empty plan fails the run here.

    ``run_id`` defaults to a unique ``{safe(job_id)}-{suffix}`` (same scheme as
    :meth:`RunStore.create`) so two runs of one spec can never collide.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    agent = agent or SourceDiscoveryAgent()
    job_id = spec.job_id or spec.name

    record = load_record(work) or new_record(run_id=run_id or fresh_run_id(job_id), job_id=job_id)
    save_spec(work, spec)
    record = save_record(work, record)

    state = load_state(work, job_id) if resume else JobState(job_id=job_id)

    # ---- 1.0 discovery (LLM-backed when the agent's provider is; metered) ----
    plan = agent.discover(spec, work_dir=work)

    if usage is not None:
        total = usage.total
        state.metrics = state.metrics.add_tokens(total.total_tokens).add_cost(total.cost_usd)
    save_state(work, state)
    save_plan(work, plan)

    # ---- pre-approval breaker check: discovery spend over the ceiling.
    # Only the two spend limits gate here; pages/storage/runtime caps are
    # acquisition concerns and still gate every fetch inside resume_run. A usable
    # plan is parked for review (the spend already happened — discarding it would
    # waste paid discovery); an empty plan has nothing worth reviewing -> FAILED.
    trip = CircuitBreaker(spec.limits).check(state.metrics)
    if trip is not None and trip.reason in ("max_llm_tokens", "max_cost_usd"):
        if plan.sources or plan.synthesis_aspects:
            state.notes = list(state.notes) + [
                f"circuit breaker at discovery: {trip} — plan kept for review"
            ]
            save_state(work, state)
            record = save_record(work, replace(record, status=RunStatus.AWAITING_APPROVAL))
            return PlannedRun(record=record, plan=plan, work_dir=work)
        state.notes = list(state.notes) + [f"circuit breaker at discovery: {trip}"]
        state.exit_reason = trip.reason
        save_state(work, state)
        record = save_record(
            work, replace(record, status=RunStatus.FAILED, exit_reason=trip.reason, error=str(trip))
        )
        return PlannedRun(record=record, plan=plan, work_dir=work)

    record = save_record(work, replace(record, status=RunStatus.AWAITING_APPROVAL))
    return PlannedRun(record=record, plan=plan, work_dir=work)


# --------------------------------------------------------------------- phase 2
def resume_run(
    work_dir: str | Path,
    decision: ApprovalDecision,
    *,
    channels: dict[SourceChannel, Channel] | None = None,
    store: CorpusStore | None = None,
    resume: bool = True,
    max_items_per_source: int = 50,
    should_cancel: Callable[[], bool] | None = None,
    clock_seconds: Callable[[], float] = time.monotonic,
    wallclock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> EngineAResult:
    """Apply ``decision`` to the persisted plan and run acquisition to artifacts.

    Loads ``spec.json`` + ``plan.json`` written by :func:`plan_run` (raises
    ``FileNotFoundError`` if either is missing), re-checks the circuit breaker
    before any fetch (discovery tokens already count), and reuses the per-source
    breaker + saturation stop + ``state.mark_done`` resume machinery.
    ``should_cancel`` is polled between sources; when it fires the loop stops with
    exit reason ``cancelled`` and the record lands at CANCELLED — an operator's
    cancel also wins over a racing terminal write.
    """
    work = Path(work_dir)
    spec = load_spec(work)
    if spec is None:
        raise FileNotFoundError(f"no {work / 'spec.json'} — run plan_run() first")
    plan = load_plan(work)
    if plan is None:
        raise FileNotFoundError(f"no {work / 'plan.json'} — run plan_run() first")

    record = load_record(work)
    if record is None:
        # never fabricate: a missing record means broken/corrupted history
        raise FileNotFoundError(f"no {work / RUN_FILE} — run plan_run() first")
    record = save_record(work, replace(record, status=RunStatus.ACQUIRING))

    try:
        result = _acquire(
            spec,
            plan,
            decision,
            work=work,
            channels=channels or {},
            store=store or LocalCorpusStore(work / "corpus"),
            resume=resume,
            max_items_per_source=max_items_per_source,
            should_cancel=should_cancel,
            clock_seconds=clock_seconds,
            wallclock=wallclock,
        )
    except Exception as exc:
        save_record(
            work,
            replace(
                record, status=RunStatus.FAILED, exit_reason="acquisition_error", error=str(exc)
            ),
        )
        raise

    current = load_record(work)
    cancelled_in_flight = current is not None and current.status is RunStatus.CANCELLED
    if result.exit_reason == "cancelled" or cancelled_in_flight:
        terminal = RunStatus.CANCELLED  # operator cancel beats the outcome
    else:
        terminal = (
            RunStatus.SUCCEEDED if result.exit_reason in SUCCESS_EXIT_REASONS else RunStatus.FAILED
        )
    save_record(work, replace(current or record, status=terminal, exit_reason=result.exit_reason))
    return result


# --------------------------------------------------------------------- composed
def run_engine_a(
    spec: JobSpec,
    *,
    work_dir: str | Path,
    agent: DiscoveryAgent | None = None,
    channels: dict[SourceChannel, Channel] | None = None,
    approver: Approver | None = None,
    store: CorpusStore | None = None,
    resume: bool = True,
    max_items_per_source: int = 50,
    clock_seconds: Callable[[], float] = time.monotonic,
    wallclock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    usage: UsageTracker | None = None,
) -> EngineAResult:
    """Run Engine A for ``spec`` and return the corpus + hand-offs + run report.

    Thin ``plan_run`` + approve + ``resume_run`` composition — one call, no pause.
    """
    planned = plan_run(spec, work_dir=work_dir, agent=agent, usage=usage, resume=resume)

    resolved_approver = approver or AutoApprover()
    if spec.discovery.approval_gate and approver is None:
        planned.plan.notes.append(
            "approval_gate on but no approver supplied -> auto-approving (non-interactive)"
        )
    decision = resolved_approver.review(planned.plan)

    return resume_run(
        work_dir,
        decision,
        channels=channels,
        store=store,
        resume=resume,
        max_items_per_source=max_items_per_source,
        clock_seconds=clock_seconds,
        wallclock=wallclock,
    )


# --------------------------------------------------------------------- acquisition
def _acquire(
    spec: JobSpec,
    plan: FetchPlan,
    decision: ApprovalDecision,
    *,
    work: Path,
    channels: dict[SourceChannel, Channel],
    store: CorpusStore,
    resume: bool,
    max_items_per_source: int,
    should_cancel: Callable[[], bool] | None = None,
    clock_seconds: Callable[[], float],
    wallclock: Callable[[], datetime],
) -> EngineAResult:
    """The acquisition loop + artifacts (the body of the pre-split run_engine_a)."""
    job_id = spec.job_id or spec.name
    breaker = CircuitBreaker(spec.limits)

    manifest_path = work / "manifest.jsonl"
    lineage_path = work / "lineage.json"
    summary_path = work / "run_summary.json"

    # plan_run persisted the (possibly fresh) state, so always read it back here:
    # it carries the discovery token/cost spend even on a non-resumed run.
    state = load_state(work, job_id)
    notes: list[str] = list(state.notes)

    # ---- prior chunks (resume): reconstruct from an existing manifest ----
    chunks: list[CorpusChunk] = []
    if resume and manifest_path.exists():
        chunks = _load_prior_chunks(manifest_path, store)
        if chunks:
            notes.append(f"resumed: {len(chunks)} chunk(s) from prior manifest")

    approved = apply_decision(plan, decision)
    notes.extend(approved.notes)  # plan notes + the approval audit line

    # ---- acquisition ----
    metrics = state.metrics
    start = clock_seconds()
    # Coverage "Brain": channels report acquired content here; it owns the
    # saturation stop signal so no novelty logic leaks into the crawler.
    governor = _SaturationGovernor()
    noted_saturated: set[str] = set()
    exit_reason: str | None = None

    fetch_ctx = FetchContext(
        job_id=job_id,
        store=store,
        spec=spec,
        max_items_per_source=max_items_per_source,
        clock=wallclock,
        governor=governor,
        staging_root=work / "media_staging",
    )

    for source in approved.sources:
        sid = source.source_id
        if state.is_done(sid):
            continue  # resume: already fetched

        if should_cancel is not None and should_cancel():
            exit_reason = "cancelled"
            notes.append("cancelled by operator between sources")
            break

        elapsed = clock_seconds() - start
        trip = breaker.check(
            metrics, storage_mb=_storage_mb(store, job_id), elapsed_seconds=elapsed
        )
        if trip is not None:
            exit_reason = trip.reason
            notes.append(f"circuit breaker: {trip}")
            break

        aspect_id = source.candidate.aspect_id
        # 1.4 coverage-saturation: skip remaining frontier sources (web crawl /
        # media downloads) for a saturated aspect
        if source.channel in _SATURATION_CHANNELS and governor.should_stop(aspect_id):
            notes.append(f"skip {sid}: aspect {aspect_id!r} already saturated")
            continue

        channel = channels.get(source.channel)
        if channel is None:
            notes.append(f"skip {sid}: no channel configured for {source.channel.value!r}")
            continue

        # The channel reports acquired content to the governor as it fetches (and
        # flushes its own frontier mid-crawl once an aspect saturates).
        fetched = channel.fetch(source, fetch_ctx)
        chunks.extend(fetched)
        metrics = metrics.add_pages(len(fetched))
        state.mark_done(sid)

        # note the first time an aspect crosses the saturation line
        if (
            source.channel in _SATURATION_CHANNELS
            and governor.should_stop(aspect_id)
            and aspect_id not in noted_saturated
        ):
            noted_saturated.add(aspect_id)
            tracker = governor.trackers[aspect_id]
            notes.append(
                f"aspect {aspect_id!r} saturated after {tracker.chunks_seen} chunk(s) "
                f"(confidence={tracker.confidence:.2f})"
            )

        save_state(work, state)

    # ---- 1.7 synthesis hand-offs ----
    handoffs = (
        handoffs_for(job_id, approved.synthesis_aspects) if decision.approve_synthesis else []
    )

    # ---- exit reason ----
    if exit_reason is None:
        if approved.is_empty:
            exit_reason = "no_sources"
        else:
            exit_reason = "completed"
    state.exit_reason = exit_reason
    state.metrics = metrics
    state.notes = notes
    save_state(work, state)

    # ---- artifacts: manifest + lineage + summary ----
    _write_manifest(manifest_path, chunks, handoffs)
    lineage = build_lineage(job_id, chunks, exit_reason=exit_reason)
    write_lineage(lineage_path, lineage)
    _write_summary(summary_path, spec, chunks, handoffs, metrics, exit_reason, plan, approved)

    return EngineAResult(
        job_id=job_id,
        plan=plan,
        approved_plan=approved,
        chunks=chunks,
        handoffs=handoffs,
        metrics=metrics,
        exit_reason=exit_reason,
        work_dir=work,
        manifest_path=manifest_path,
        lineage_path=lineage_path,
        summary_path=summary_path,
        notes=notes,
    )


# --------------------------------------------------------------------- helpers
class _SaturationGovernor:
    """Coverage Brain backing :class:`FetchContext` (OUTLINE §1.4).

    One :class:`SaturationTracker` per aspect. Channels call :meth:`observe` as they
    acquire content; :meth:`should_stop` reports saturation so the web frontier is
    flushed mid-crawl and remaining same-aspect sources are skipped. Saturation
    logic lives here, in the orchestrator — never in the crawler.
    """

    def __init__(self) -> None:
        self.trackers: dict[str, SaturationTracker] = {}

    def observe(self, aspect_id: str, text: str) -> None:
        self.trackers.setdefault(aspect_id, SaturationTracker()).update(text)

    def should_stop(self, aspect_id: str) -> bool:
        tracker = self.trackers.get(aspect_id)
        return tracker is not None and tracker.is_saturated()


def _storage_mb(store: CorpusStore, job_id: str) -> float:
    size = getattr(store, "size_mb", None)
    return size(job_id) if callable(size) else 0.0


def _load_prior_chunks(manifest_path: Path, store: CorpusStore) -> list[CorpusChunk]:
    out: list[CorpusChunk] = []
    for n, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            # a torn/corrupt manifest means provenance is broken — surface it
            raise ValueError(f"corrupt {manifest_path.name} line {n}: {exc}") from exc
        if row.get("kind") != "chunk":
            continue
        preview = ""
        uri = row.get("content_uri", "")
        if uri and store.exists(uri):
            preview = store.get(uri).strip()[:280]
        out.append(CorpusChunk.from_manifest_row(row, text_preview=preview))
    return out


def _write_manifest(
    path: Path, chunks: list[CorpusChunk], handoffs: list[SynthesisHandoff]
) -> None:
    rows = [c.to_manifest_row() for c in chunks]
    rows += [{**h.to_manifest_row(), "kind": "handoff"} for h in handoffs]
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")


def _write_summary(
    path: Path,
    spec: JobSpec,
    chunks: list[CorpusChunk],
    handoffs: list[SynthesisHandoff],
    metrics: RunMetrics,
    exit_reason: str,
    plan: FetchPlan,
    approved: FetchPlan,
) -> None:
    by_channel: dict[str, int] = {}
    for c in chunks:
        by_channel[c.source_channel.value] = by_channel.get(c.source_channel.value, 0) + 1
    summary = {
        "job_id": spec.job_id or spec.name,
        "name": spec.name,
        "exit_reason": exit_reason,
        "metrics": metrics.to_dict(),
        "n_chunks": len(chunks),
        "chunks_by_channel": by_channel,
        "n_synthesis_handoffs": len(handoffs),
        "n_aspects": len(plan.aspects),
        "discovered_sources": len(plan.sources),
        "approved_sources": len(approved.sources),
        "total_tokens_in_corpus": sum(c.n_tokens for c in chunks),
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
