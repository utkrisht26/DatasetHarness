"""Tests for the engine phase split: plan_run -> (approval pause) -> resume_run.

Reuses the offline agents/channels from test_orchestration; the composed
run_engine_a regression suite lives there and must stay green (CRITICAL).
"""

from __future__ import annotations

import copy

import pytest

from adh.contract import JobSpec
from adh.discovery import ApprovalDecision, AutoApprover
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


def test_plan_run_over_token_budget_fails_before_the_gate(minimal_spec_dict, tmp_path):
    d = copy.deepcopy(minimal_spec_dict)
    d["limits"] = {"max_llm_tokens": 100}
    spec = JobSpec.model_validate(d)
    usage = UsageTracker()
    usage.add(TokenUsage(input_tokens=90, output_tokens=20))  # 110 >= 100

    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent(), usage=usage)
    assert not planned.awaiting_approval
    assert planned.record.status is RunStatus.FAILED
    assert planned.record.exit_reason == "max_llm_tokens"


def test_plan_run_surfaced_cost_over_budget_fails(minimal_spec_dict, tmp_path):
    d = copy.deepcopy(minimal_spec_dict)
    d["limits"] = {"max_cost_usd": 1.0}
    spec = JobSpec.model_validate(d)
    usage = UsageTracker()
    usage.add(TokenUsage(input_tokens=10, output_tokens=5, cost_usd=2.5))

    planned = plan_run(spec, work_dir=tmp_path, agent=_mixed_agent(), usage=usage)
    assert planned.record.status is RunStatus.FAILED
    assert planned.record.exit_reason == "max_cost_usd"


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
    web_ids = frozenset(
        s.source_id for s in planned.plan.sources if s.channel.value == "web"
    )
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
    assert planned.record.status is RunStatus.FAILED  # plan_run already flags it

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
