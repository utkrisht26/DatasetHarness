"""Tests for the ``adh`` CLI (validate / discover / run) and the interactive gate."""

from __future__ import annotations

import json
import os

import pytest

from adh import cli
from adh.settings import DISCOVERY_MODEL_ENV
from adh.discovery.models import (
    Aspect,
    CandidateSource,
    FetchPlan,
    RankedSource,
    SourceEvaluation,
)
from adh.contract import SourceChannel


def test_cli_validate_ok(example_spec_path, capsys):
    rc = cli.main(["validate", str(example_spec_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "validation OK" in out
    assert "filesystem checks OK" in out


def test_cli_never_loads_the_developer_dotenv_under_test(example_spec_path):
    # main() loads the repo .env; under test that handed real provider keys and
    # model ladders to the code under test (live, billed calls that could hang).
    before = dict(os.environ)
    assert cli.main(["validate", str(example_spec_path)]) == 0
    assert dict(os.environ) == before


def test_cli_discover_prints_plan(example_spec_path, capsys):
    rc = cli.main(["discover", str(example_spec_path), "--no-llm"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "Fetch plan" in out
    assert "discovery LLM spend: 0 token(s)" in out


class _PlanOnlyAgent:
    def discover(self, spec, *, work_dir=None):
        return _plan()


@pytest.mark.parametrize(
    ("argv_tail", "env_model", "expected"),
    [
        (["--discovery-model", "openai:flag-model"], None, "openai:flag-model"),
        ([], "openai:env-model", "openai:env-model"),
        (["--no-llm"], "openai:env-model", None),
    ],
)
def test_cli_discover_hands_the_configured_model_to_discovery(
    example_spec_path, capsys, monkeypatch, argv_tail, env_model, expected
):
    if env_model:
        monkeypatch.setenv(DISCOVERY_MODEL_ENV, env_model)
    seen: list[str | None] = []

    def fake_factory(model):
        seen.append(model)
        return lambda _usage: _PlanOnlyAgent()

    monkeypatch.setattr(cli, "live_agent_factory", fake_factory)

    rc = cli.main(["discover", str(example_spec_path), *argv_tail])

    assert rc == 0
    assert seen == [expected]
    assert "Fetch plan for job 'j1'" in capsys.readouterr().out


def test_cli_run_no_fetch_autonomous(example_spec_path, tmp_path, capsys, monkeypatch):
    # hermetic: no provider keys even when a developer's .env was loaded
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    rc = cli.main(
        [
            "run",
            str(example_spec_path),
            "--work-dir",
            str(tmp_path),
            "--autonomous",
            "--no-fetch",
            "--no-llm",
        ]
    )
    out = capsys.readouterr().out
    assert rc == 0
    assert "Engine A finished" in out
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert "exit_reason" in summary


def _plan() -> FetchPlan:
    cand = CandidateSource(
        source_id="a0:web:https://x",
        channel=SourceChannel.WEB,
        locator="https://x",
        title="X",
        aspect_id="a0",
    )
    ev = SourceEvaluation("a0:web:https://x", 0.5, 0.5, True, True, 5, 0.5)
    return FetchPlan(
        job_id="j1",
        sources=(RankedSource(cand, ev),),
        synthesis_aspects=(Aspect("a1", "t", "r", ("k",)),),
    )


def test_interactive_decision_all(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "a")
    decision = cli._interactive_decision(_plan())
    assert "a0:web:https://x" in decision.approved_source_ids
    assert decision.approve_synthesis


def test_interactive_decision_none(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "n")
    decision = cli._interactive_decision(_plan())
    assert decision.approved_source_ids == frozenset()
    assert not decision.approve_synthesis


def test_interactive_decision_subset(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "1")
    decision = cli._interactive_decision(_plan())
    assert decision.approved_source_ids == frozenset({"a0:web:https://x"})


def test_cli_requires_subcommand():
    with pytest.raises(SystemExit):
        cli.main([])
