"""Tests for the engine phase split: plan_run -> (approval pause) -> resume_run.

Reuses the offline agents/channels from test_orchestration; the composed
run_engine_a regression suite lives there and must stay green (CRITICAL).
"""

from __future__ import annotations

import copy

import pytest

from adh.contract import JobSpec
from adh.discovery import ApprovalDecision, AutoApprover, FetchPlan
from adh.llm import TokenUsage, UsageTracker
from adh.orchestration import RunStatus, plan_run, resume_run
from adh.orchestration.run_store import load_plan, load_record, load_spec
from adh.orchestration.state import load_state

from .test_orchestration import _mixed_agent, _mixed_channels, _mixed_spec


def _approve_all(plan) -> ApprovalDecision:
    return AutoApprover().review(plan)


# ------------------------------------------------------------------ plan_run
def test_plan_run_persists_plan_and_awaits_approval(minimal_spec_dict, tmp_path):
    spec = _mixed_spec(minimal_spec_dict)
    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent())

    assert planned.awaiting_approval
    assert planned.record.status is RunStatus.AWAITING_APPROVAL
    assert planned.plan.sources  # discovery found the canned sources
    # everything needed to survive a restart is on disk
    assert load_spec(tmp_path) == spec
    assert load_plan(tmp_path) == planned.plan
    assert load_record(tmp_path).status is RunStatus.AWAITING_APPROVAL


def test_plan_run_folds_discovery_usage_into_metrics(minimal_spec_dict, tmp_path):
    spec = _mixed_spec(minimal_spec_dict)
    usage = UsageTracker()
    usage.add(TokenUsage(input_tokens=1200, output_tokens=300))
    plan_run(spec, work_dir=tmp_path, agent=_mixed_agent(), usage=usage)

    metrics = load_state(tmp_path, spec.name).metrics
    assert metrics.llm_tokens == 1500
    assert metrics.cost_usd is None  # no provider-surfaced cost -> unknown


def test_plan_run_over_token_budget_parks_usable_plan_for_review(minimal_spec_dict, tmp_path):
    """Discovery spend over the token ceiling no longer discards paid-for work:
    a usable plan is parked at the approval gate with the breaker noted."""
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns. Tokio runtime internals scheduler."
    d["motivation"] = "teach async"
    d["limits"] = {"max_llm_tokens": 100}
    spec = JobSpec.model_validate(d)
    usage = UsageTracker()
    usage.add(TokenUsage(input_tokens=90, output_tokens=20))  # 110 >= 100

    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent(), usage=usage)
    assert planned.awaiting_approval
    assert planned.plan.sources  # the paid-for plan is kept, not discarded
    assert planned.record.status is RunStatus.AWAITING_APPROVAL
    state = load_state(tmp_path, spec.name)
    assert any("circuit breaker at discovery" in n for n in state.notes)


def test_plan_run_over_token_budget_without_sources_fails(minimal_spec_dict, tmp_path):
    """Over budget AND nothing discovered -> FAILED; there is no plan to save."""

    class _EmptyAgent:
        def discover(self, spec, work_dir):
            return FetchPlan(job_id=spec.job_id or spec.name)

    d = copy.deepcopy(minimal_spec_dict)
    d["limits"] = {"max_llm_tokens": 100}
    spec = JobSpec.model_validate(d)
    usage = UsageTracker()
    usage.add(TokenUsage(input_tokens=90, output_tokens=20))  # 110 >= 100

    planned = plan_run(spec, work_dir=tmp_path, agent=_EmptyAgent(), usage=usage)
    assert not planned.awaiting_approval
    assert planned.record.status is RunStatus.FAILED
    assert planned.record.exit_reason == "max_llm_tokens"


def test_plan_run_surfaced_cost_over_budget_parks_usable_plan(minimal_spec_dict, tmp_path):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns. Tokio runtime internals scheduler."
    d["motivation"] = "teach async"
    d["limits"] = {"max_cost_usd": 1.0}
    spec = JobSpec.model_validate(d)
    usage = UsageTracker()
    usage.add(TokenUsage(input_tokens=10, output_tokens=5, cost_usd=2.5))

    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent(), usage=usage)
    assert planned.awaiting_approval  # usable plan kept for review, breaker noted
    assert planned.plan.sources
    assert any("circuit breaker at discovery" in n for n in load_state(tmp_path, spec.name).notes)


def test_plan_run_pages_cap_does_not_fail_discovery(minimal_spec_dict, tmp_path):
    """Acquisition caps are not discovery failures; the loop still gates fetches."""
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns. Tokio runtime internals scheduler."
    d["motivation"] = "teach async"
    d["limits"] = {"max_pages_scraped": 0}
    spec = JobSpec.model_validate(d)
    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent())
    assert planned.awaiting_approval  # parked, not failed


# ------------------------------------------------------------------ resume_run
def test_resume_run_without_plan_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        resume_run(tmp_path, ApprovalDecision(frozenset()))


def test_resume_run_applies_a_subset_decision(minimal_spec_dict, tmp_path):
    spec = _mixed_spec(minimal_spec_dict)
    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent())
    web_ids = frozenset(s.source_id for s in planned.plan.sources if s.channel.value == "web")
    assert web_ids and len(web_ids) < len(planned.plan.sources)

    result = resume_run(
        tmp_path,
        ApprovalDecision(web_ids, approve_synthesis=False, reason="web only"),
        channels=_mixed_channels(),
    )
    assert result.exit_reason == "completed"
    assert {c.source_channel.value for c in result.chunks} == {"web"}
    assert result.n_handoffs == 0
    assert load_record(tmp_path).status is RunStatus.SUCCEEDED


def test_resume_run_metrics_carry_discovery_tokens(minimal_spec_dict, tmp_path):
    """max_llm_tokens trips pre-acquisition: discovery spend gates the first fetch."""
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns. Tokio runtime internals scheduler."
    d["motivation"] = "teach async"
    d["limits"] = {"max_llm_tokens": 2000}
    spec = JobSpec.model_validate(d)
    usage = UsageTracker()
    usage.add(TokenUsage(input_tokens=1900, output_tokens=200))  # 2100 >= 2000

    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent(), usage=usage)
    # over-budget discovery parks (usable plan), but the breaker still holds
    assert planned.record.status is RunStatus.AWAITING_APPROVAL

    # Even if a caller resumes anyway, the per-source breaker holds the ceiling.
    result = resume_run(tmp_path, _approve_all(planned.plan), channels=_mixed_channels())
    assert result.exit_reason == "max_llm_tokens"
    assert result.n_chunks == 0
    assert load_record(tmp_path).status is RunStatus.FAILED


def test_resume_run_marks_failed_on_acquisition_error(minimal_spec_dict, tmp_path):
    spec = _mixed_spec(minimal_spec_dict)
    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent())

    class _Boom:
        def fetch(self, source, ctx):
            raise RuntimeError("channel exploded")

    from adh.contract import SourceChannel

    with pytest.raises(RuntimeError, match="channel exploded"):
        resume_run(
            tmp_path,
            _approve_all(planned.plan),
            channels={SourceChannel.DATASETS: _Boom(), SourceChannel.WEB: _Boom()},
        )
    record = load_record(tmp_path)
    assert record.status is RunStatus.FAILED
    assert record.exit_reason == "acquisition_error"
    assert "channel exploded" in record.error


def test_phase_split_survives_restart(minimal_spec_dict, tmp_path):
    """The pause is durable: a fresh process resumes purely from disk."""
    spec = _mixed_spec(minimal_spec_dict)
    plan_run(spec, work_dir=tmp_path, agent=_mixed_agent())

    # "restart": no in-memory state — rebuild the decision from the persisted plan
    plan = load_plan(tmp_path)
    assert plan is not None
    result = resume_run(tmp_path, _approve_all(plan), channels=_mixed_channels())
    assert result.exit_reason == "completed"
    assert result.n_chunks > 0


# ------------------------------------------------------------------ hygiene
def test_plan_run_generates_unique_run_ids(minimal_spec_dict, tmp_path):
    """Two runs of the same spec never collide on run_id (the CLI path)."""
    spec = _mixed_spec(minimal_spec_dict)
    first = plan_run(spec, work_dir=tmp_path / "a", agent=_mixed_agent())
    second = plan_run(spec, work_dir=tmp_path / "b", agent=_mixed_agent())
    assert first.record.run_id != second.record.run_id
    assert first.record.run_id.startswith("demo-")  # safe(job_id) prefix kept


def test_resume_run_never_fabricates_a_record(minimal_spec_dict, tmp_path):
    """A missing run.json is a loud error, not a silently reset history."""
    spec = _mixed_spec(minimal_spec_dict)
    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent())
    (tmp_path / "run.json").unlink()
    with pytest.raises(FileNotFoundError, match="run.json"):
        resume_run(tmp_path, _approve_all(planned.plan), channels=_mixed_channels())


def test_mid_flight_cancel_stops_between_sources(minimal_spec_dict, tmp_path):
    """An operator cancel during acquisition halts the loop at the next source."""
    from adh.contract import SourceChannel

    spec = _mixed_spec(minimal_spec_dict)
    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent())
    assert len(planned.plan.sources) >= 2

    flag = {"cancelled": False}

    class _CancelAfterFirst:
        def fetch(self, source, ctx):
            flag["cancelled"] = True  # the operator hits cancel during fetch 1
            return []

    channel = _CancelAfterFirst()
    result = resume_run(
        tmp_path,
        _approve_all(planned.plan),
        channels={SourceChannel.DATASETS: channel, SourceChannel.WEB: channel},
        should_cancel=lambda: flag["cancelled"],
    )
    assert result.exit_reason == "cancelled"
    assert load_record(tmp_path).status is RunStatus.CANCELLED
    state = load_state(tmp_path, spec.name)
    assert any("cancelled" in n for n in state.notes)


def test_corrupt_manifest_line_fails_loudly(minimal_spec_dict, tmp_path):
    """A torn/corrupt manifest is surfaced with file+line, never silently skipped."""
    spec = _mixed_spec(minimal_spec_dict)
    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent())
    resume_run(tmp_path, _approve_all(planned.plan), channels=_mixed_channels())
    # corrupt the durable manifest, then resume again (the prior-chunks path)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text('{corrupt json\n{"kind": "chunk", "chunk_id": "ok"}\n')

    with pytest.raises(ValueError, match="manifest.jsonl"):
        resume_run(tmp_path, _approve_all(planned.plan), channels=_mixed_channels())


def test_corrupt_state_file_warns_loudly(minimal_spec_dict, tmp_path, caplog):
    """A torn state.json starts fresh (read paths must survive) but announces the
    loss — silently resetting it would quietly lift the breaker ceilings."""
    import logging

    from adh.orchestration.state import JobState, save_state

    spec = _mixed_spec(minimal_spec_dict)
    save_state(tmp_path, JobState(job_id=spec.name, completed_source_ids={"src:1"}))
    (tmp_path / "state.json").write_text("{torn json", encoding="utf-8")

    with caplog.at_level(logging.WARNING, logger="adh.orchestration.state"):
        state = load_state(tmp_path, spec.name)

    assert state.completed_source_ids == set()  # fresh, but the loss was logged
    assert any("state.json" in rec.message for rec in caplog.records)


def test_state_job_id_mismatch_warns(minimal_spec_dict, tmp_path, caplog):
    """A state.json belonging to another job is not silently adopted either."""
    import logging

    from adh.orchestration.state import JobState, save_state

    spec = _mixed_spec(minimal_spec_dict)
    save_state(tmp_path, JobState(job_id="some-other-job"))

    with caplog.at_level(logging.WARNING, logger="adh.orchestration.state"):
        state = load_state(tmp_path, spec.name)

    assert state.job_id == spec.name
    assert any(rec.levelno >= logging.WARNING for rec in caplog.records)
