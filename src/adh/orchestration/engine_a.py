"""Engine A driver — discovery -> approval -> acquisition -> corpus + manifest.

Ties the Phase 1 pieces into one resumable, limit-bounded run:

    discover (1.0) -> approval gate (1.0) -> per-channel fetch (1.1/1.6/1.7)
        with coverage-saturation stop (1.4) and the circuit-breaker floor (1.4)
        -> offloaded corpus (1.3) + manifest + lineage (3.3) + durable state (1.5)

The function is *mechanism*: the caller chooses the approver (the CLI prompts a human
when ``approval_gate`` is on; tests inject a policy). Synthesis-routed aspects bypass
acquisition and are recorded as hand-offs to Engine B.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from adh.channels import FetchContext, handoffs_for
from adh.channels.base import Channel
from adh.contract import JobSpec, SourceChannel
from adh.corpus import CorpusChunk, CorpusStore, LocalCorpusStore, SynthesisHandoff
from adh.discovery import (
    Approver,
    AutoApprover,
    FetchPlan,
    SourceDiscoveryAgent,
    apply_decision,
)
from adh.provenance import build_lineage, write_lineage

from .limits import CircuitBreaker, RunMetrics
from .saturation import SaturationTracker
from .state import JobState, load_state, save_state


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


def run_engine_a(
    spec: JobSpec,
    *,
    work_dir: str | Path,
    agent: SourceDiscoveryAgent | None = None,
    channels: dict[SourceChannel, Channel] | None = None,
    approver: Approver | None = None,
    store: CorpusStore | None = None,
    resume: bool = True,
    max_items_per_source: int = 50,
    clock_seconds: Callable[[], float] = time.monotonic,
    wallclock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> EngineAResult:
    """Run Engine A for ``spec`` and return the corpus + hand-offs + run report."""
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    agent = agent or SourceDiscoveryAgent()
    channels = channels or {}
    store = store or LocalCorpusStore(work / "corpus")
    breaker = CircuitBreaker(spec.limits)

    job_id = spec.job_id or spec.name
    manifest_path = work / "manifest.jsonl"
    lineage_path = work / "lineage.json"
    summary_path = work / "run_summary.json"

    state = load_state(work, job_id) if resume else JobState(job_id=job_id)
    notes: list[str] = list(state.notes)

    # ---- prior chunks (resume): reconstruct from an existing manifest ----
    chunks: list[CorpusChunk] = []
    if resume and manifest_path.exists():
        chunks = _load_prior_chunks(manifest_path, store)
        if chunks:
            notes.append(f"resumed: {len(chunks)} chunk(s) from prior manifest")

    # ---- 1.0 discovery ----
    plan = agent.discover(spec)
    notes.extend(plan.notes)

    # ---- 1.0 approval gate (default ON; caller supplies the approver) ----
    resolved_approver = approver or AutoApprover()
    if spec.discovery.approval_gate and approver is None:
        notes.append("approval_gate on but no approver supplied -> auto-approving (non-interactive)")
    decision = resolved_approver.review(plan)
    approved = apply_decision(plan, decision)

    # ---- acquisition ----
    metrics = state.metrics
    start = clock_seconds()
    saturation: dict[str, SaturationTracker] = {}
    saturated_aspects: set[str] = set()
    exit_reason: str | None = None

    fetch_ctx = FetchContext(
        job_id=job_id,
        store=store,
        spec=spec,
        max_items_per_source=max_items_per_source,
        clock=wallclock,
    )

    for source in approved.sources:
        sid = source.source_id
        if state.is_done(sid):
            continue  # resume: already fetched

        elapsed = clock_seconds() - start
        trip = breaker.check(metrics, storage_mb=_storage_mb(store, job_id), elapsed_seconds=elapsed)
        if trip is not None:
            exit_reason = trip.reason
            notes.append(f"circuit breaker: {trip}")
            break

        aspect_id = source.candidate.aspect_id
        # 1.4 coverage-saturation: skip remaining WEB sources for a saturated aspect
        if source.channel == SourceChannel.WEB and aspect_id in saturated_aspects:
            notes.append(f"skip {sid}: aspect {aspect_id!r} already saturated")
            continue

        channel = channels.get(source.channel)
        if channel is None:
            notes.append(f"skip {sid}: no channel configured for {source.channel.value!r}")
            continue

        fetched = channel.fetch(source, fetch_ctx)
        chunks.extend(fetched)
        metrics = metrics.add_pages(len(fetched))
        state.mark_done(sid)

        # feed the saturation tracker for web aspects
        if source.channel == SourceChannel.WEB and fetched:
            tracker = saturation.setdefault(aspect_id, SaturationTracker())
            for ch in fetched:
                tracker.update(_chunk_text(ch, store))
            if tracker.is_saturated():
                saturated_aspects.add(aspect_id)
                notes.append(
                    f"aspect {aspect_id!r} saturated after {tracker.chunks_seen} chunk(s) "
                    f"(confidence={tracker.confidence:.2f})"
                )

        save_state(work, state)

    # ---- 1.7 synthesis hand-offs ----
    handoffs = handoffs_for(job_id, approved.synthesis_aspects) if decision.approve_synthesis else []

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
def _storage_mb(store: CorpusStore, job_id: str) -> float:
    size = getattr(store, "size_mb", None)
    return size(job_id) if callable(size) else 0.0


def _chunk_text(chunk: CorpusChunk, store: CorpusStore) -> str:
    try:
        return store.get(chunk.content_uri)
    except (OSError, FileNotFoundError):
        return chunk.text_preview


def _load_prior_chunks(manifest_path: Path, store: CorpusStore) -> list[CorpusChunk]:
    out: list[CorpusChunk] = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
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
