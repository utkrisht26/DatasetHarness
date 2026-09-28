"""Engine B driver — end-to-end over an Engine A run directory, offline."""

from __future__ import annotations

import json

import pytest

from adh.contract import JobSpec, JobSpecError
from adh.generation import HeuristicGenerator
from adh.orchestration import run_engine_a, run_engine_b
from adh.orchestration.engine_b import load_corpus
from adh.verification import HeuristicJudge


def _run_b(spec, work, **kw):
    """Offline Engine B: inject the deterministic collaborators (no model / no key)."""
    kw.setdefault("generator", HeuristicGenerator())
    kw.setdefault("judge", HeuristicJudge())
    return run_engine_b(spec, work_dir=work, **kw)


_GOAL = "Rust async macro patterns and examples."
_URL = "https://tokio.rs/async-macros"


def _spec(minimal_spec_dict, **over) -> JobSpec:
    data = {**minimal_spec_dict, "description": _GOAL, "motivation": "teach async macros", **over}
    return JobSpec(**data)


def _seed_corpus(spec, work) -> None:
    """Run Engine A (offline heuristic) to lay down corpus + manifest for Engine B."""
    from adh.channels.web import InMemoryContentFetcher, WebChannel
    from adh.contract import SourceChannel
    from adh.discovery import SourceDiscoveryAgent
    from adh.search import InMemoryDatasetRegistry, InMemorySearchProvider, SearchHit

    search = InMemorySearchProvider(
        pages=[
            SearchHit(
                url=_URL,
                title="Rust async macro patterns and examples",
                snippet="rust async macro patterns examples expand at compile time",
                published="2024-02-01",
                license_hint="MIT",
            )
        ]
    )
    fetcher = InMemoryContentFetcher(
        pages={
            _URL: "Async macros in Rust expand at compile time. They reduce boilerplate "
            "and keep call sites readable across many practical examples and patterns."
        }
    )
    agent = SourceDiscoveryAgent(search=search, registry=InMemoryDatasetRegistry())
    channels = {SourceChannel.WEB: WebChannel(fetcher=fetcher)}
    run_engine_a(spec, work_dir=work, agent=agent, channels=channels)


# --------------------------------------------------------------------- tier 2 (soft, heuristic jury)
def test_engine_b_end_to_end_soft(minimal_spec_dict, tmp_path):
    spec = _spec(minimal_spec_dict, target_size={"records": 12})
    _seed_corpus(spec, tmp_path)

    result = _run_b(spec, tmp_path)

    assert result.exit_reason in ("completed", "no_records")
    assert result.n_records > 0
    # every emitted record validates against the contract + schema (build_record did it)
    for rec in result.records:
        assert rec.input and rec.output
        assert rec.split in ("train", "validation", "test")
        assert rec.judge_scores is not None  # tier-2 records carry jury scores
    # dataset shards + manifest + docs on disk
    assert (result.dataset_dir / "manifest.jsonl").exists()
    assert (result.dataset_dir / "datasheet.md").exists()
    assert (result.dataset_dir / "croissant.json").exists()
    assert result.delivery.total_records == result.n_records
    assert result.summary_path.exists()


def test_engine_b_records_are_deduplicated(minimal_spec_dict, tmp_path):
    spec = _spec(minimal_spec_dict, target_size={"records": 20})
    _seed_corpus(spec, tmp_path)
    result = _run_b(spec, tmp_path)
    cluster_ids = [r.dedup_cluster_id for r in result.records]
    # shipped records are representatives: no two share a cluster, none flagged duplicate
    assert len(cluster_ids) == len(set(cluster_ids))
    assert all(r.is_duplicate is False for r in result.records)


# --------------------------------------------------------------------- tier 1 (hard, real verifier)
@pytest.mark.parametrize(
    ("schema_ref", "expected"),
    [
        ("http://schema.org/Text", "local JSON Schema"),
        ("./schemas/qa_record.schema.json", "x-verifier"),
        ("./schemas/missing.schema.json", "not found"),
    ],
)
def test_invalid_build_config_spends_nothing_and_preserves_artifacts(
    minimal_spec_dict, tmp_path, monkeypatch, schema_ref, expected
):
    from adh.orchestration import engine_b

    spec = _spec(
        minimal_spec_dict,
        verifiability_tier=1,
        judge_config=None,
        record_schema_ref=schema_ref,
    )

    def unexpected(*args, **kwargs):
        pytest.fail("invalid build must not construct a model")

    monkeypatch.setattr(engine_b, "build_generator", unexpected)
    monkeypatch.setattr(engine_b, "build_judge", unexpected)
    before = {}
    for name in ("engine_b_state.json", "engine_b_verified.jsonl", "manifest.jsonl"):
        (tmp_path / name).write_text("prior artifact")
        before[name] = (tmp_path / name).read_bytes()

    with pytest.raises(JobSpecError, match=expected):
        run_engine_b(spec, work_dir=tmp_path, resume=False)

    assert {name: (tmp_path / name).read_bytes() for name in before} == before
    assert not (tmp_path / "dataset").exists()


def test_build_summary_explains_rejected_records(minimal_spec_dict, tmp_path, caplog):
    from adh.verification.jury import PanelResult

    class RejectingJudge:
        def evaluate(self, **kwargs):
            return PanelResult(scores={"accuracy": 1.0}, models=("test:judge",))

    spec = _spec(minimal_spec_dict, target_size={"records": 2})
    _seed_corpus(spec, tmp_path)
    result = _run_b(spec, tmp_path, judge=RejectingJudge())
    assert result.n_records == 0
    summary = json.loads(result.summary_path.read_text())
    assert any("verification rejection (jury): jury overall" in n for n in summary["notes"])
    assert "rejected at jury: jury overall" in caplog.text


def test_engine_b_tier1_code_uses_real_verifier(minimal_spec_dict, tmp_path):
    spec = _spec(
        minimal_spec_dict,
        verifiability_tier=1,
        modality="code",
        record_schema_ref="./schemas/code_record.schema.json",
        judge_config=None,
        task_categories=["text-generation"],
        target_size={"records": 8},
    )
    _seed_corpus(spec, tmp_path)
    result = _run_b(spec, tmp_path)
    assert result.n_records > 0
    for rec in result.records:
        # tier-1 records carry a passing deterministic verifier result, no jury
        assert rec.verifier_result is not None and rec.verifier_result.passed is True
        assert rec.judge_scores is None


# --------------------------------------------------------------------- circuit breaker + resume
def test_engine_b_circuit_breaker_floor(minimal_spec_dict, tmp_path):
    spec = _spec(minimal_spec_dict, target_size={"records": 20})
    spec.limits.max_runtime_seconds = 0  # floor: do not start any generation
    _seed_corpus(spec, tmp_path)
    result = _run_b(spec, tmp_path, clock_seconds=lambda: 100.0)
    assert result.exit_reason == "max_runtime_seconds"
    assert result.n_records == 0


def test_engine_b_resume_reuses_verified(minimal_spec_dict, tmp_path):
    spec = _spec(minimal_spec_dict, target_size={"records": 12})
    _seed_corpus(spec, tmp_path)
    first = _run_b(spec, tmp_path)
    assert (tmp_path / "engine_b_verified.jsonl").exists()
    # a second run resumes: all seeds already completed -> same accepted count, no regen
    second = _run_b(spec, tmp_path)
    assert second.n_records == first.n_records
    assert any("resumed" in n for n in second.notes)


# --------------------------------------------------------------------- stranded-resume recovery
def test_engine_b_resume_regenerates_stranded_seeds(minimal_spec_dict, tmp_path):
    """A prior build that marked seeds complete but generated nothing (provider outage)
    must not be re-skipped forever — resume clears the stranded bookkeeping and rebuilds."""
    spec = _spec(minimal_spec_dict, target_size={"records": 4})
    _seed_corpus(spec, tmp_path)
    (tmp_path / "engine_b_state.json").write_text(
        json.dumps(
            {
                "job_id": spec.job_id or spec.name,
                "completed_seed_ids": ["stranded:a0:x:0", "stranded:a1:y:1"],
                "metrics": {},
                "counters": {},
                "notes": [],
            }
        )
    )

    result = _run_b(spec, tmp_path)

    assert result.n_records > 0
    assert any("generated nothing" in n for n in result.notes)
    state = json.loads((tmp_path / "engine_b_state.json").read_text())
    assert state["completed_seed_ids"]  # this build re-marked the regenerated seeds


def test_engine_b_resume_keeps_completed_seeds_with_generation_history(minimal_spec_dict, tmp_path):
    """Bookkeeping backed by real generation (counters.generated > 0) is honest work:
    resume still skips those seeds even when no candidate survived curation."""
    spec = _spec(minimal_spec_dict, target_size={"records": 4})
    _seed_corpus(spec, tmp_path)
    first = _run_b(spec, tmp_path)
    assert first.n_records > 0
    # simulate "everything generated was later dropped": keep bstate, drop the journal
    (tmp_path / "engine_b_verified.jsonl").unlink()

    second = _run_b(spec, tmp_path)

    assert second.n_records == 0
    assert not any("generated nothing" in n for n in second.notes)


def test_engine_b_persists_counters_through_total_generation_failure(minimal_spec_dict, tmp_path):
    """An all-seeds-failed build leaves evidence on disk — counters land in bstate and
    the summary names the exhaustion instead of exiting 'no_records' in silence."""
    spec = _spec(minimal_spec_dict, target_size={"records": 3})
    _seed_corpus(spec, tmp_path)

    class ExplodingGenerator:
        def generate(self, seed, *, spec, store=None):
            raise RuntimeError("provider down")

    result = _run_b(spec, tmp_path, generator=ExplodingGenerator())

    assert result.n_records == 0
    state = json.loads((tmp_path / "engine_b_state.json").read_text())
    assert state["counters"]["generation_errors"] >= 1
    summary = json.loads((tmp_path / "engine_b_summary.json").read_text())
    assert summary["counts"]["generation_errors"] >= 1


def test_engine_b_logs_generation_progress(minimal_spec_dict, tmp_path, caplog):
    """A multi-hour build must be observable: announce the work, then log each seed."""
    import logging

    spec = _spec(minimal_spec_dict, target_size={"records": 4})
    _seed_corpus(spec, tmp_path)

    with caplog.at_level(logging.INFO, logger="adh.orchestration.engine_b"):
        _run_b(spec, tmp_path)

    msgs = [r.message for r in caplog.records]
    assert any("seed(s) to generate" in m for m in msgs)
    progress = [m for m in msgs if "engine B progress" in m]
    assert progress  # at least one per-seed line


# --------------------------------------------------------------------- corpus loader
def test_load_corpus_reads_manifest(minimal_spec_dict, tmp_path):
    from adh.corpus import LocalCorpusStore

    spec = _spec(minimal_spec_dict, target_size={"records": 6})
    _seed_corpus(spec, tmp_path)
    chunks, handoffs = load_corpus(tmp_path, LocalCorpusStore(tmp_path / "corpus"))
    assert chunks  # web channel produced at least one chunk
    assert all(c.source_url for c in chunks)


# --------------------------------------------------------------------- quality report integration
def test_engine_b_attaches_quality_report(minimal_spec_dict, tmp_path):
    """Every build ships an offline quality card: summary carries it, and a
    markdown copy lands beside the dataset for the publish flow to upload."""
    spec = _spec(minimal_spec_dict, target_size={"records": 12})
    _seed_corpus(spec, tmp_path)

    result = _run_b(spec, tmp_path)

    assert result.quality_report is not None
    q = result.quality_report
    assert q.total_records == result.n_records
    assert 0.0 <= q.duplicate_rate <= 1.0
    summary = json.loads((tmp_path / "engine_b_summary.json").read_text())
    assert summary["quality"]["total_records"] == result.n_records
    md = tmp_path / "dataset" / "quality_report.md"
    assert md.exists() and "| total_records |" in md.read_text()
