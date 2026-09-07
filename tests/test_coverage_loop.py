"""Coverage-gap loop driver — thin Engine B aspects feed back into discovery.

The driver composes the two engines into OUTLINE §1.0's loop: build -> measure
per-aspect coverage -> re-discover sources for thin aspects -> acquire only the
new sources -> rebuild. Every test here runs fully offline (heuristic generator
+ judge fallbacks, in-memory channels).
"""

from __future__ import annotations

import copy
import json
import shutil

import pytest

from adh.channels import (
    DatasetsChannel,
    InMemoryContentFetcher,
    InMemoryDatasetLoader,
    WebChannel,
)
from adh.contract import JobSpec, SourceChannel
from adh.discovery import AutoApprover, SourceDiscoveryAgent
from adh.discovery.models import Aspect, CandidateSource, RankedSource, SourceEvaluation
from adh.generation import HeuristicGenerator
from adh.orchestration import run_coverage_loop, run_engine_a
from adh.verification import HeuristicJudge
from adh.search import (
    DatasetHit,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
)


# ----------------------------------------------------------------- world
RUST_DS = DatasetHit(
    dataset_id="org/rust-async-macros-qa",
    locator="https://huggingface.co/datasets/org/rust-async-macros-qa",
    title="Rust async macro Q/A",
    description="rust async macro design declarative procedural question-answering input output",
    license="MIT",
    downloads=5000,
    last_modified="2024-09-01",
)

# Distinct grounding paragraphs: one per fresh page, so each new source yields a
# distinct chunk text (and therefore a distinct generated output that survives
# per-shard dedup).
_PARAGRAPHS = {
    1: "ownership boundaries pin the executor state machine and waker latency budget",
    2: "the runtime multiplexes tasks across worker threads with work stealing queues",
    3: "cancellation safety hinges on drop semantics racing the poll deadline clock",
}


def _spec(minimal_spec_dict, records: int = 18) -> JobSpec:
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns. Tokio runtime internals scheduler."
    d["motivation"] = "teach async"
    # Small target => the fair-share threshold is a handful of records; a single
    # split means dedup survivors == distinct outputs (deterministic closure math).
    d["target_size"] = {"records": records}
    d["splits"] = {"train": 1.0}
    return JobSpec.model_validate(d)


def _initial_agent() -> SourceDiscoveryAgent:
    """Round-1 discovery: one dataset source, no web hits -> at least one thin aspect."""
    return SourceDiscoveryAgent(
        registry=InMemoryDatasetRegistry(
            datasets=[RUST_DS],
            samples={
                RUST_DS.dataset_id: "rust async macro declarative procedural input output examples"
            },
        ),
        search=InMemorySearchProvider(),
    )


def _initial_channels() -> dict:
    loader = InMemoryDatasetLoader(
        records={"org/rust-async-macros-qa": ["Q: macro? A: macro_rules!", "Q: async? A: future"]}
    )
    return {SourceChannel.DATASETS: DatasetsChannel(loader=loader)}


def _ranked(source_id: str, url: str, title: str, aspect_id: str) -> RankedSource:
    return RankedSource(
        candidate=CandidateSource(
            source_id=source_id,
            channel=SourceChannel.WEB,
            locator=url,
            title=title,
            aspect_id=aspect_id,
            snippet=title[:120],
            license="MIT",
            discovered_via="coverage-loop",
        ),
        evaluation=SourceEvaluation(
            source_id=source_id,
            shape_match=0.9,
            quality=0.9,
            license_ok=True,
            freshness_ok=True,
            estimated_yield=3,
            score=0.9,
        ),
    )


class _GapAgent:
    """Rediscover port double: three fresh distinct web pages per thin aspect."""

    def __init__(self):
        self.rediscovered_aspects: list[str] = []

    def rediscover(self, spec, aspect: Aspect, *, existing):
        self.rediscovered_aspects.append(aspect.aspect_id)
        fresh = [
            _ranked(
                f"{aspect.aspect_id}-deep-{i}",
                f"https://docs.example/{aspect.aspect_id}/deep-{i}",
                f"{aspect.topic} deep dive part {i}",
                aspect.aspect_id,
            )
            for i in (1, 2, 3)
        ]
        return type(existing)(
            job_id=existing.job_id,
            sources=tuple([*existing.sources, *fresh]),
            synthesis_aspects=existing.synthesis_aspects,
            aspects=existing.aspects,
            notes=[
                *existing.notes,
                f"re-discovery for aspect {aspect.aspect_id!r}: +{len(fresh)} candidate(s)",
            ],
        )


class _DryAgent:
    """Rediscover port double that never finds anything new."""

    def rediscover(self, spec, aspect, *, existing):
        return existing


class _GuardedFetcher:
    """Serves the loop's fresh pages; fetching anything else is an incremental bug."""

    def __init__(self):
        self._inner = InMemoryContentFetcher(
            pages={
                f"https://docs.example/{aid}/deep-{i}": (
                    f"Aspect {aid} deep dive part {i}: {_PARAGRAPHS[i]} "
                    + " ".join(f"{aid}{i}{n}" for n in range(12))
                )
                for aid in (f"a{n}" for n in range(8))
                for i in (1, 2, 3)
            }
        )

    def fetch(self, url, *, max_pages=1):
        if url not in self._inner.pages:
            raise AssertionError(f"refetched a non-fresh locator: {url}")
        return self._inner.fetch(url, max_pages=max_pages)


class _GuardedLoader:
    def __init__(self):
        self.calls: list[str] = []

    def load(self, dataset_id, *, limit):
        self.calls.append(dataset_id)
        raise AssertionError(f"refetched completed dataset source: {dataset_id}")


def _loop_channels() -> dict:
    return {
        SourceChannel.WEB: WebChannel(fetcher=_GuardedFetcher()),
        SourceChannel.DATASETS: DatasetsChannel(loader=_GuardedLoader()),
    }


def _manifest_chunk_rows(tmp_path) -> int:
    manifest = tmp_path / "manifest.jsonl"
    if not manifest.exists():
        return 0
    rows = [json.loads(x) for x in manifest.read_text().splitlines() if x.strip()]
    return sum(1 for r in rows if r.get("kind") == "chunk")


def _prepare_engine_a(spec, tmp_path):
    result = run_engine_a(
        spec, work_dir=tmp_path, agent=_initial_agent(), channels=_initial_channels(),
        approver=AutoApprover(),
    )
    assert result.exit_reason == "completed"
    return result


# ----------------------------------------------------------------- happy path
def test_gap_closes_after_one_rediscovery_round(minimal_spec_dict, tmp_path):
    spec = _spec(minimal_spec_dict)
    _prepare_engine_a(spec, tmp_path)
    baseline_chunks = _manifest_chunk_rows(tmp_path)

    gap_agent = _GapAgent()
    outcome = run_coverage_loop(
        spec,
        work_dir=tmp_path,
        agent=gap_agent,
        channels=_loop_channels(),
        approver=AutoApprover(),
        generator=HeuristicGenerator(),
        judge=HeuristicJudge(),
        max_rounds=2,
    )

    assert outcome.exit_reason == "coverage_closed"
    assert outcome.rounds == 1
    assert outcome.result.coverage.thin_aspects == ()
    assert outcome.thin_aspects == ()
    assert outcome.acquisitions and outcome.acquisitions[0].exit_reason == "completed"
    assert gap_agent.rediscovered_aspects  # at least one thin aspect drove rediscovery
    assert _manifest_chunk_rows(tmp_path) > baseline_chunks  # fresh corpus landed
    plan_text = (tmp_path / "plan.json").read_text()
    assert "re-discovery" in plan_text  # merged plan persisted for the audit trail


# ----------------------------------------------------------------- termination guards
def test_no_new_sources_stops_the_loop(minimal_spec_dict, tmp_path):
    spec = _spec(minimal_spec_dict)
    _prepare_engine_a(spec, tmp_path)

    outcome = run_coverage_loop(
        spec, work_dir=tmp_path, agent=_DryAgent(), channels=_loop_channels(),
        generator=HeuristicGenerator(),
        judge=HeuristicJudge(),
    )

    assert outcome.exit_reason == "no_new_sources"
    assert outcome.rounds == 0
    assert outcome.acquisitions == []
    assert any("no new sources" in n for n in outcome.notes)


def test_max_rounds_zero_never_rediscovering(minimal_spec_dict, tmp_path):
    spec = _spec(minimal_spec_dict)
    _prepare_engine_a(spec, tmp_path)

    outcome = run_coverage_loop(
        spec, work_dir=tmp_path, agent=_GapAgent(), channels=_loop_channels(),
        max_rounds=0, generator=HeuristicGenerator(), judge=HeuristicJudge(),
    )

    assert outcome.exit_reason == "max_rounds"
    assert outcome.rounds == 0
    assert outcome.acquisitions == []


def test_cancel_check_runs_between_rounds(minimal_spec_dict, tmp_path):
    spec = _spec(minimal_spec_dict)
    _prepare_engine_a(spec, tmp_path)

    checks = {"n": 0}

    def should_cancel():
        checks["n"] += 1
        return checks["n"] > 1  # allow the first round, cancel before the second

    outcome = run_coverage_loop(
        spec,
        work_dir=tmp_path,
        agent=_GapAgent(),
        channels=_loop_channels(),
        max_rounds=3,
        generator=HeuristicGenerator(),
        judge=HeuristicJudge(),
        should_cancel=should_cancel,
    )
    assert outcome.exit_reason == "cancelled"
    assert outcome.rounds == 1


# ----------------------------------------------------------------- contract
def test_requires_an_engine_a_work_dir(minimal_spec_dict, tmp_path):
    spec = _spec(minimal_spec_dict)
    with pytest.raises(ValueError, match="[Ee]ngine A"):
        run_coverage_loop(
            spec, work_dir=tmp_path, agent=_GapAgent(), channels={},
            generator=HeuristicGenerator(),
            judge=HeuristicJudge(),
        )


# ----------------------------------------------------------------- spec coherence
def test_spec_divergence_is_resaved_before_the_loop(minimal_spec_dict, tmp_path):
    """resume_run reloads spec.json — the loop must re-save the caller's spec so
    re-acquisition enforces the same rules discovery/build already used."""
    spec = _spec(minimal_spec_dict)
    _prepare_engine_a(spec, tmp_path)

    drifted = _spec(minimal_spec_dict, records=99)
    assert drifted != spec

    outcome = run_coverage_loop(
        drifted, work_dir=tmp_path, agent=_GapAgent(), channels=_loop_channels(),
        max_rounds=0, generator=HeuristicGenerator(), judge=HeuristicJudge(),
    )

    assert outcome.exit_reason == "max_rounds"
    on_disk = json.loads((tmp_path / "spec.json").read_text())
    assert on_disk["target_size"]["records"] == 99  # one shared contract everywhere
    assert any("differed" in n for n in outcome.notes)


# ----------------------------------------------------------------- terminal branches
def test_rejection_at_round_approval_stops_reacquisition(minimal_spec_dict, tmp_path):
    """A non-completed re-acquisition folds whatever arrived into a rebuilt dataset,
    then surfaces ``reacquisition_stopped`` instead of looping forever."""
    from adh.discovery import ApprovalDecision

    class _Rejector:
        def review(self, plan):
            return ApprovalDecision(frozenset(), approve_synthesis=False, reason="test: none")

    spec = _spec(minimal_spec_dict)
    _prepare_engine_a(spec, tmp_path)

    outcome = run_coverage_loop(
        spec, work_dir=tmp_path, agent=_GapAgent(), channels=_loop_channels(),
        approver=_Rejector(), max_rounds=2,
        generator=HeuristicGenerator(), judge=HeuristicJudge(),
    )

    assert outcome.exit_reason == "reacquisition_stopped"
    assert outcome.acquisitions[-1].exit_reason == "no_sources"
    assert any("stopped early" in n for n in outcome.notes)
    assert outcome.result.n_records >= 0  # the fold-back rebuild ran


def test_mixed_dry_and_fresh_aspects_are_both_handled(minimal_spec_dict, tmp_path):
    """One thin aspect yields nothing while another yields fresh sources: the dry
    one is noted, the fresh one is acquired — neither silently dropped."""
    big_target = 200  # fair-share threshold exceeds any offline yield -> all thin

    class _MixedAgent:
        """Dries on the first thin aspect asked, feeds fresh pages afterwards."""

        def __init__(self):
            self.seen: list[str] = []
            self.dried_yet = False

        def rediscover(self, spec, aspect, *, existing):
            self.seen.append(aspect.aspect_id)
            if not self.dried_yet:
                self.dried_yet = True
                return existing
            return _GapAgent().rediscover(spec, aspect, existing=existing)

    spec = _spec(minimal_spec_dict, records=big_target)
    _prepare_engine_a(spec, tmp_path)

    agent = _MixedAgent()
    baseline_chunks = _manifest_chunk_rows(tmp_path)

    outcome = run_coverage_loop(
        spec, work_dir=tmp_path, agent=agent, channels=_loop_channels(),
        approver=AutoApprover(), max_rounds=1,
        generator=HeuristicGenerator(), judge=HeuristicJudge(),
    )

    assert len(agent.seen) >= 2, "world must ask about several thin aspects"
    dry_id = agent.seen[0]
    assert any(dry_id in n and "nothing new" in n for n in outcome.notes)
    assert outcome.rounds == 1
    assert _manifest_chunk_rows(tmp_path) > baseline_chunks  # fresh aspect still landed


# ----------------------------------------------------------------- CLI wiring
def test_cli_loop_runs_the_driver(repo_root, tmp_path, capsys, monkeypatch):
    """`adh loop` composes the driver with live pieces; hermetic under offline flags."""
    import yaml

    from adh import cli

    # A spec without an explicit pii_policy: the contract default degrades loudly
    # instead of fail-closing when the presidio extra is absent (offline CI).
    spec_dict = {
        "name": "demo",
        "description": "Rust async macro patterns. Tokio runtime internals scheduler.",
        "motivation": "teach async",
        "intended_uses": ["supervised-fine-tuning"],
        "verifiability_tier": 2,
        "task_categories": ["question-answering"],
        "languages": ["en"],
        "target_size": {"records": 12},
        "record_schema_ref": "./schemas/qa_record.schema.json",
        "allowed_licenses": ["MIT", "Apache-2.0"],
        "generator_models": [],
        "judge_config": {
            "models": ["anthropic:claude-sonnet-4-6", "openai:gpt-judge"],
            "min_score": 3.5,
        },
        "decontaminate_against": ["./eval/heldout_rust_qa.jsonl"],
        "output_license": "MIT",
        "splits": {"train": 1.0},
    }
    spec_path = tmp_path / "specs" / "loop-spec.yaml"
    spec_path.parent.mkdir()
    spec_path.write_text(yaml.safe_dump(spec_dict), encoding="utf-8")
    # record_schema_ref / decontaminate_against resolve against the spec's base
    # dir (spec at <tmp>/specs/... -> base <tmp>, mirroring examples/ in-repo)
    (tmp_path / "schemas").mkdir()
    shutil.copy(repo_root / "schemas" / "qa_record.schema.json", tmp_path / "schemas")
    (tmp_path / "eval").mkdir()
    shutil.copy(repo_root / "eval" / "heldout_rust_qa.jsonl", tmp_path / "eval")

    # The live heuristic agent samples real network endpoints during discovery
    # (slow, non-hermetic) — swap in the same in-memory world the driver tests use.
    monkeypatch.setattr(
        cli,
        "live_agent_factory",
        lambda _model=None: (
            lambda _usage: SourceDiscoveryAgent(
                registry=InMemoryDatasetRegistry(
                    datasets=[RUST_DS],
                    samples={
                        RUST_DS.dataset_id: "rust async macro declarative procedural "
                        "input output examples"
                    },
                ),
                search=InMemorySearchProvider(),
            )
        ),
    )
    # hermetic even when a developer's shell/.env carries provider keys
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert cli.main(["run", str(spec_path), "--work-dir", str(tmp_path / "work"),
                     "--autonomous", "--no-fetch", "--no-llm"]) == 0
    assert cli.main(["loop", str(spec_path), "--work-dir", str(tmp_path / "work"),
                     "--max-rounds", "1", "--autonomous", "--no-fetch", "--no-llm"]) == 0
    out = capsys.readouterr().out
    assert "coverage loop" in out.lower()
    assert "exit_reason=" in out
