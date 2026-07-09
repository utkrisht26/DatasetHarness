"""Runs router — Engine A behind the approval gate, over HTTP (T6).

:class:`RunsService` owns the run state machine on top of the file-based
:class:`RunStore` and a small worker pool:

    POST /runs            -> DISCOVERING on a worker (plan_run)
                             -> AWAITING_APPROVAL (parked, no thread held)
                             -> autonomous specs auto-approve inline instead
    POST /runs/{id}/approve -> ACQUIRING on a worker (resume_run) -> terminal
    POST /runs/{id}/cancel  -> CANCELLED (only before acquisition starts)

Everything is injectable for offline tests: the discovery agent factory (fed a
per-run :class:`UsageTracker` for metering) and the acquisition channels.
"""

from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from adh.contract import JobSpec
from adh.discovery import ApprovalDecision, AutoApprover, DiscoveryAgent, FetchPlan
from adh.generation import Generator
from adh.llm import UsageTracker
from adh.orchestration import RunRecord, RunStatus, RunStore, plan_run, resume_run, run_engine_b
from adh.orchestration.state import load_state
from adh.verification import JudgePanel

logger = logging.getLogger(__name__)

# agent_factory receives the run's UsageTracker so discovery spend is metered.
AgentFactory = Callable[[UsageTracker], DiscoveryAgent]


class RunConflictError(RuntimeError):
    """The requested transition is invalid for the run's current status (HTTP 409)."""


class RunsService:
    """Owns the run store, the worker pool, and the run state machine."""

    def __init__(
        self,
        base_dir: str | Path,
        *,
        agent_factory: AgentFactory | None = None,
        channels: dict | None = None,
        workers: int = 2,
        max_items_per_source: int = 50,
        engine_b_generator: Generator | None = None,
        engine_b_judge: JudgePanel | None = None,
    ) -> None:
        self.store = RunStore(base_dir)
        self._agent_factory = agent_factory or _default_agent
        self._channels = channels or {}
        self._max_items_per_source = max_items_per_source
        # Engine B collaborators: None -> run_engine_b picks the live LLM path
        # (metered per run); inject deterministic ones for offline tests.
        self._eb_generator = engine_b_generator
        self._eb_judge = engine_b_judge
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="run")
        self._lock = threading.Lock()
        self._cancelled: set[str] = set()

    # ------------------------------------------------------------------ queries
    def get(self, run_id: str) -> RunRecord | None:
        return self.store.get(run_id)

    def list(self) -> list[RunRecord]:
        return self.store.list()

    # ------------------------------------------------------------------ actions
    def submit(self, spec: JobSpec) -> RunRecord:
        record = self.store.create(spec)
        self._executor.submit(self._plan, record.run_id)
        return record

    def approve(self, run_id: str, decision: ApprovalDecision) -> RunRecord:
        with self._lock:
            record = self.store.get(run_id)
            if record is None:
                raise KeyError(run_id)
            if record.status is not RunStatus.AWAITING_APPROVAL:
                raise RunConflictError(
                    f"run {run_id!r} is {record.status.value!r}, not awaiting approval"
                )
            record = self.store.set_status(run_id, RunStatus.ACQUIRING)
        self._executor.submit(self._acquire, run_id, decision)
        return record

    def build(self, run_id: str) -> RunRecord:
        """Run Engine B (synthesis + curation) over a succeeded run, on a worker."""
        with self._lock:
            record = self.store.get(run_id)
            if record is None:
                raise KeyError(run_id)
            if record.status is not RunStatus.SUCCEEDED:
                raise RunConflictError(
                    f"run {run_id!r} is {record.status.value!r}; only a succeeded run can be built"
                )
        self._executor.submit(self._build_dataset, run_id)
        return record

    def dataset_summary(self, run_id: str) -> dict | None:
        """The Engine B run summary if a dataset has been built, else None."""
        path = self.store.run_dir(run_id) / "engine_b_summary.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):  # pragma: no cover - torn read
            return None

    def cancel(self, run_id: str) -> RunRecord:
        with self._lock:
            record = self.store.get(run_id)
            if record is None:
                raise KeyError(run_id)
            if record.status not in (RunStatus.DISCOVERING, RunStatus.AWAITING_APPROVAL):
                raise RunConflictError(
                    f"run {run_id!r} is {record.status.value!r}; only a run that has "
                    "not started acquiring can be cancelled"
                )
            self._cancelled.add(run_id)
            return self.store.set_status(run_id, RunStatus.CANCELLED)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        for channel in self._channels.values():
            close = getattr(channel, "close", None)
            if callable(close):
                close()

    # ------------------------------------------------------------------ workers
    def _plan(self, run_id: str) -> None:
        spec = self.store.load_spec(run_id)
        if spec is None:  # pragma: no cover - create() always writes the spec
            self.store.set_status(run_id, RunStatus.FAILED, error="spec.json missing")
            return
        usage = UsageTracker()
        agent = self._agent_factory(usage)
        logger.info("run %s: discovery starting (job=%s)", run_id, spec.job_id or spec.name)
        try:
            planned = plan_run(
                spec, work_dir=self.store.run_dir(run_id), agent=agent, usage=usage
            )
        except Exception as exc:  # noqa: BLE001 - a failed run must not kill the worker
            logger.exception(
                "run %s: discovery failed after %d token(s)", run_id, usage.total.total_tokens
            )
            self.store.set_status(run_id, RunStatus.FAILED, error=str(exc))
            return
        logger.info(
            "run %s: discovery done — %d source(s), status=%s, %d token(s)",
            run_id,
            len(planned.plan.sources),
            planned.record.status.value,
            usage.total.total_tokens,
        )

        with self._lock:
            if run_id in self._cancelled:
                # cancel() raced plan_run's awaiting-approval write; cancellation wins
                self.store.set_status(run_id, RunStatus.CANCELLED)
                return
        if planned.record.status is RunStatus.FAILED:
            return  # discovery over budget; plan_run already recorded it
        if not spec.discovery.approval_gate:
            # autonomous: no pause — auto-approve and acquire on this same worker
            decision = AutoApprover().review(planned.plan)
            self.store.set_status(run_id, RunStatus.ACQUIRING)
            self._acquire(run_id, decision)

    def _acquire(self, run_id: str, decision: ApprovalDecision) -> None:
        logger.info(
            "run %s: acquiring %d approved source(s)",
            run_id,
            len(decision.approved_source_ids),
        )
        try:
            resume_run(
                self.store.run_dir(run_id),
                decision,
                channels=self._channels,
                max_items_per_source=self._max_items_per_source,
            )
        except Exception:  # noqa: BLE001 - resume_run recorded FAILED; keep the worker alive
            logger.exception("acquisition failed for run %s", run_id)
        else:
            logger.info("run %s: acquisition finished", run_id)

    def _build_dataset(self, run_id: str) -> None:
        """Worker: run Engine B over a succeeded run's corpus into a dataset."""
        spec = self.store.load_spec(run_id)
        if spec is None:  # pragma: no cover - a succeeded run always has its spec
            logger.error("run %s: cannot build — spec.json missing", run_id)
            return
        usage = UsageTracker()
        logger.info("run %s: dataset build starting (job=%s)", run_id, spec.job_id or spec.name)
        try:
            result = run_engine_b(
                spec,
                work_dir=self.store.run_dir(run_id),
                generator=self._eb_generator,
                judge=self._eb_judge,
                usage=usage,
            )
        except Exception:  # noqa: BLE001 - a failed build must not kill the worker
            logger.exception(
                "run %s: dataset build failed after %d token(s)", run_id, usage.total.total_tokens
            )
            return
        logger.info(
            "run %s: dataset build done — %d record(s), exit=%s, %d token(s)",
            run_id,
            result.n_records,
            result.exit_reason,
            usage.total.total_tokens,
        )


def _default_agent(usage: UsageTracker) -> DiscoveryAgent:
    """Offline-safe default: the heuristic agent (the gateway wires richer ones)."""
    from adh.discovery import SourceDiscoveryAgent

    return SourceDiscoveryAgent()


# ------------------------------------------------------------------ API models
class ApprovalRequest(BaseModel):
    """POST body for the approve endpoint. Omitted ids -> approve every source."""

    model_config = ConfigDict(extra="forbid")

    approved_source_ids: list[str] | None = None
    approve_synthesis: bool = True
    reason: str = ""


class MetricsOut(BaseModel):
    pages_scraped: int = 0
    llm_tokens: int = 0
    cost_usd: float | None = None  # null = provider surfaced no cost ("unknown")


class PlanAspectOut(BaseModel):
    aspect_id: str
    topic: str
    rationale: str = ""
    keywords: list[str] = Field(default_factory=list)


class PlanSourceOut(BaseModel):
    source_id: str
    channel: str
    locator: str
    title: str
    aspect_id: str
    snippet: str = ""
    license: str | None = None
    published: str | None = None
    discovered_via: str = ""
    score: float
    shape_match: float
    quality: float
    license_ok: bool
    freshness_ok: bool
    estimated_yield: int
    notes: str = ""


class PlanOut(BaseModel):
    job_id: str
    sources: list[PlanSourceOut] = Field(default_factory=list)
    synthesis_aspects: list[PlanAspectOut] = Field(default_factory=list)
    aspects: list[PlanAspectOut] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class RunSummaryOut(BaseModel):
    run_id: str
    job_id: str
    status: str
    created_at: str
    updated_at: str
    exit_reason: str | None = None
    error: str | None = None


class RunOut(RunSummaryOut):
    plan: PlanOut | None = None  # present once discovery has produced one
    metrics: MetricsOut = Field(default_factory=MetricsOut)
    notes: list[str] = Field(default_factory=list)


def _summary(record: RunRecord) -> RunSummaryOut:
    return RunSummaryOut(
        run_id=record.run_id,
        job_id=record.job_id,
        status=record.status.value,
        created_at=record.created_at,
        updated_at=record.updated_at,
        exit_reason=record.exit_reason,
        error=record.error,
    )


def _plan_out(plan: FetchPlan) -> PlanOut:
    def aspect_out(a) -> PlanAspectOut:
        return PlanAspectOut(
            aspect_id=a.aspect_id,
            topic=a.topic,
            rationale=a.rationale,
            keywords=list(a.keywords),
        )

    return PlanOut(
        job_id=plan.job_id,
        sources=[
            PlanSourceOut(
                source_id=s.source_id,
                channel=s.channel.value,
                locator=s.candidate.locator,
                title=s.candidate.title,
                aspect_id=s.candidate.aspect_id,
                snippet=s.candidate.snippet,
                license=s.candidate.license,
                published=s.candidate.published,
                discovered_via=s.candidate.discovered_via,
                score=s.evaluation.score,
                shape_match=s.evaluation.shape_match,
                quality=s.evaluation.quality,
                license_ok=s.evaluation.license_ok,
                freshness_ok=s.evaluation.freshness_ok,
                estimated_yield=s.evaluation.estimated_yield,
                notes=s.evaluation.notes,
            )
            for s in plan.sources
        ],
        synthesis_aspects=[aspect_out(a) for a in plan.synthesis_aspects],
        aspects=[aspect_out(a) for a in plan.aspects],
        notes=list(plan.notes),
    )


# ------------------------------------------------------------------ the router
def build_runs_router(runs: RunsService) -> APIRouter:
    router = APIRouter(tags=["runs"])

    @router.post("", status_code=202)
    def submit_run(spec: JobSpec) -> RunSummaryOut:
        return _summary(runs.submit(spec))

    @router.get("")
    def list_runs() -> list[RunSummaryOut]:
        return [_summary(r) for r in runs.list()]

    @router.get("/{run_id}")
    def get_run(run_id: str) -> RunOut:
        record = runs.get(run_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        plan = runs.store.load_plan(run_id)
        state = load_state(runs.store.run_dir(run_id), record.job_id)
        return RunOut(
            **_summary(record).model_dump(),
            plan=_plan_out(plan) if plan is not None else None,
            metrics=MetricsOut(**state.metrics.to_dict()),
            notes=list(state.notes),
        )

    @router.post("/{run_id}/approve")
    def approve_run(run_id: str, body: ApprovalRequest) -> RunSummaryOut:
        if runs.get(run_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}")
        plan = runs.store.load_plan(run_id)
        if plan is None:
            raise HTTPException(
                status_code=409, detail=f"run {run_id!r} has no plan to approve yet"
            )
        plan_ids = {s.source_id for s in plan.sources}
        if body.approved_source_ids is None:
            approved = plan_ids
        else:
            approved = set(body.approved_source_ids)
            unknown = sorted(approved - plan_ids)
            if unknown:
                raise HTTPException(
                    status_code=422, detail=f"unknown source ids: {unknown}"
                )
        decision = ApprovalDecision(
            approved_source_ids=frozenset(approved),
            approve_synthesis=body.approve_synthesis,
            reason=body.reason or "api approval",
        )
        try:
            return _summary(runs.approve(run_id, decision))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}") from None
        except RunConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    @router.post("/{run_id}/cancel")
    def cancel_run(run_id: str) -> RunSummaryOut:
        try:
            return _summary(runs.cancel(run_id))
        except KeyError:
            raise HTTPException(status_code=404, detail=f"unknown run {run_id!r}") from None
        except RunConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from None

    return router
