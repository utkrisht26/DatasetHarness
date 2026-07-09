"""Engine B driver — end-to-end over an Engine A run directory, offline."""

from __future__ import annotations



from adh.contract import JobSpec
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


# --------------------------------------------------------------------- corpus loader
def test_load_corpus_reads_manifest(minimal_spec_dict, tmp_path):
    from adh.corpus import LocalCorpusStore

    spec = _spec(minimal_spec_dict, target_size={"records": 6})
    _seed_corpus(spec, tmp_path)
    chunks, handoffs = load_corpus(tmp_path, LocalCorpusStore(tmp_path / "corpus"))
    assert chunks  # web channel produced at least one chunk
    assert all(c.source_url for c in chunks)
