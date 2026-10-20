"""Tests for the shared-spine slice: limits, saturation, state, Engine A driver."""

from __future__ import annotations

import copy
import json

import pytest

from adh.channels import (
    DatasetsChannel,
    InMemoryContentFetcher,
    InMemoryDatasetLoader,
    WebChannel,
)
from adh.contract import JobSpec, SourceChannel
from adh.discovery import AutoApprover, SourceDiscoveryAgent
from adh.orchestration import (
    CircuitBreaker,
    JobState,
    RunMetrics,
    SaturationTracker,
    load_state,
    run_engine_a,
    save_state,
)
from adh.search import (
    DatasetHit,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
    SearchHit,
)


# ----------------------------------------------------------------- metrics
def test_run_metrics_are_immutable_reducers():
    m0 = RunMetrics()
    m1 = m0.add_pages(3).add_tokens(100).add_cost(0.5)
    assert m0.pages_scraped == 0  # original unchanged
    assert m1.pages_scraped == 3 and m1.llm_tokens == 100 and m1.cost_usd == 0.5
    assert RunMetrics.from_dict(m1.to_dict()) == m1


# ----------------------------------------------------------------- breaker
@pytest.mark.parametrize(
    "kwargs,metrics,reason",
    [
        ({"max_pages_scraped": 2}, RunMetrics(pages_scraped=3), "max_pages_scraped"),
        ({"max_llm_tokens": 10}, RunMetrics(llm_tokens=11), "max_llm_tokens"),
        ({"max_cost_usd": 1.0}, RunMetrics(cost_usd=1.5), "max_cost_usd"),
    ],
)
def test_breaker_trips(minimal_spec_dict, kwargs, metrics, reason):
    d = copy.deepcopy(minimal_spec_dict)
    d["limits"] = kwargs
    spec = JobSpec.model_validate(d)
    trip = CircuitBreaker(spec.limits).check(metrics)
    assert trip is not None and trip.reason == reason


def test_breaker_storage_and_time(minimal_spec_dict):
    d = copy.deepcopy(minimal_spec_dict)
    d["limits"] = {"max_storage_mb": 1, "max_runtime_seconds": 5}
    spec = JobSpec.model_validate(d)
    b = CircuitBreaker(spec.limits)
    assert b.check(RunMetrics(), storage_mb=2).reason == "max_storage_mb"
    assert b.check(RunMetrics(), elapsed_seconds=6).reason == "max_runtime_seconds"
    assert b.check(RunMetrics(), storage_mb=0.5, elapsed_seconds=1) is None


# ----------------------------------------------------------------- saturation
def test_saturation_trips_on_repeated_content():
    t = SaturationTracker(confidence_threshold=0.7, min_chunks=3)
    for _ in range(4):
        t.update("the quick brown fox jumps over the lazy dog repeatedly today")
    assert t.is_saturated()
    assert t.confidence >= 0.7


def test_saturation_not_tripped_on_novel_content():
    t = SaturationTracker(confidence_threshold=0.7, min_chunks=3)
    t.update("alpha beta gamma delta epsilon zeta eta theta")
    t.update("one two three four five six seven eight nine ten")
    t.update("red orange yellow green blue indigo violet white black")
    assert not t.is_saturated()


# ----------------------------------------------------------------- state
def test_state_round_trip(tmp_path):
    st = JobState(job_id="j1")
    st.mark_done("s1")
    st.metrics = RunMetrics(pages_scraped=5)
    st.exit_reason = "completed"
    save_state(tmp_path, st)
    loaded = load_state(tmp_path, "j1")
    assert loaded.is_done("s1")
    assert loaded.metrics.pages_scraped == 5
    assert loaded.exit_reason == "completed"


def test_state_mismatched_job_is_fresh(tmp_path):
    save_state(tmp_path, JobState(job_id="other"))
    fresh = load_state(tmp_path, "j1")
    assert fresh.completed_source_ids == set()


# ----------------------------------------------------------------- e2e fixtures
def _mixed_spec(minimal_spec_dict) -> JobSpec:
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns. Tokio runtime internals scheduler."
    d["motivation"] = "teach async"
    return JobSpec.model_validate(d)


RUST_DS = DatasetHit(
    dataset_id="org/rust-async-macros-qa",
    locator="https://huggingface.co/datasets/org/rust-async-macros-qa",
    title="Rust async macro Q/A",
    description="rust async macro design declarative procedural question-answering input output",
    license="MIT",
    downloads=5000,
    last_modified="2024-09-01",
)


def _mixed_agent() -> SourceDiscoveryAgent:
    registry = InMemoryDatasetRegistry(
        datasets=[RUST_DS],
        samples={
            RUST_DS.dataset_id: "rust async macro declarative procedural input output examples"
        },
    )
    search = InMemorySearchProvider(
        pages=[
            SearchHit(
                url="https://tokio.rs/runtime-internals",
                title="Tokio runtime internals scheduler",
                snippet="tokio runtime internals scheduler tasks input output",
                published="2024-02-01",
                license_hint="MIT",
            )
        ]
    )
    return SourceDiscoveryAgent(registry=registry, search=search)


def _mixed_channels() -> dict:
    loader = InMemoryDatasetLoader(
        records={"org/rust-async-macros-qa": ["Q: macro? A: macro_rules!", "Q: async? A: future"]}
    )
    fetcher = InMemoryContentFetcher(
        pages={"https://tokio.rs/runtime-internals": "Tokio runtime internals: the scheduler ..."}
    )
    return {
        SourceChannel.DATASETS: DatasetsChannel(loader=loader),
        SourceChannel.WEB: WebChannel(fetcher=fetcher),
    }


# ----------------------------------------------------------------- e2e run
def test_engine_a_end_to_end(minimal_spec_dict, tmp_path):
    spec = _mixed_spec(minimal_spec_dict)
    result = run_engine_a(
        spec,
        work_dir=tmp_path,
        agent=_mixed_agent(),
        channels=_mixed_channels(),
        approver=AutoApprover(),
    )
    assert result.exit_reason == "completed"
    channels = {c.source_channel for c in result.chunks}
    assert SourceChannel.DATASETS in channels
    assert SourceChannel.WEB in channels
    # artifacts written
    assert result.manifest_path.exists()
    assert result.lineage_path.exists()
    assert result.summary_path.exists()
    summary = json.loads(result.summary_path.read_text())
    assert summary["n_chunks"] == result.n_chunks
    assert summary["exit_reason"] == "completed"
    # manifest has chunk rows
    lines = [json.loads(x) for x in result.manifest_path.read_text().splitlines() if x.strip()]
    assert any(r["kind"] == "chunk" for r in lines)


def test_engine_a_circuit_breaker_floor(minimal_spec_dict, tmp_path):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns. Tokio runtime internals scheduler."
    d["motivation"] = "teach async"
    d["limits"] = {"max_pages_scraped": 0}  # trip immediately
    spec = JobSpec.model_validate(d)
    result = run_engine_a(
        spec,
        work_dir=tmp_path,
        agent=_mixed_agent(),
        channels=_mixed_channels(),
        approver=AutoApprover(),
    )
    assert result.exit_reason == "max_pages_scraped"
    assert result.n_chunks == 0


def test_engine_a_synthesis_only(minimal_spec_dict, tmp_path):
    # No registry/web hits -> rust/async/macro are model-known -> synthesis handoffs.
    spec = _mixed_spec(minimal_spec_dict)
    agent = SourceDiscoveryAgent(
        registry=InMemoryDatasetRegistry(), search=InMemorySearchProvider()
    )
    result = run_engine_a(
        spec, work_dir=tmp_path, agent=agent, channels={}, approver=AutoApprover()
    )
    assert result.n_chunks == 0
    assert result.n_handoffs >= 1
    assert result.exit_reason == "completed"


def test_engine_a_resume_skips_completed(minimal_spec_dict, tmp_path):
    spec = _mixed_spec(minimal_spec_dict)
    agent = _mixed_agent()
    channels = _mixed_channels()
    first = run_engine_a(
        spec, work_dir=tmp_path, agent=agent, channels=channels, approver=AutoApprover()
    )
    n_first = first.n_chunks
    completed = load_state(tmp_path, spec.name).completed_source_ids
    assert completed  # state persisted

    # Second run with channels that would RAISE if fetched again — proves no refetch.
    class _ExplodingLoader:
        def load(self, dataset_id, *, limit):
            raise AssertionError("should not refetch a completed source")

    class _ExplodingFetcher:
        def fetch(self, url, *, max_pages=1):
            raise AssertionError("should not refetch a completed source")

    boom = {
        SourceChannel.DATASETS: DatasetsChannel(loader=_ExplodingLoader()),
        SourceChannel.WEB: WebChannel(fetcher=_ExplodingFetcher()),
    }
    second = run_engine_a(
        spec, work_dir=tmp_path, agent=agent, channels=boom, approver=AutoApprover()
    )
    # Resumed: prior chunks reloaded from manifest, nothing refetched.
    assert second.n_chunks == n_first
    assert second.exit_reason == "completed"


def test_engine_a_page_budget_is_shared_fairly_across_sources(minimal_spec_dict, tmp_path):
    # Live run rust-async-live-small: the first approved source deep-crawled 50 pages
    # (max_items_per_source) against max_pages_scraped=30 — overshooting the cap and
    # starving the other 23 sources. Each source must get a fair share of what's left.
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Tokio runtime internals scheduler tasks."
    d["motivation"] = "teach tokio"
    d["limits"] = {"max_pages_scraped": 6}
    spec = JobSpec.model_validate(d)

    hubs = [f"https://site{i}.dev/hub" for i in range(3)]
    pages: dict[str, str] = {}
    links: dict[str, list[str]] = {}
    for i, hub in enumerate(hubs):
        children = [f"https://site{i}.dev/page{j}" for j in range(10)]
        pages[hub] = f"site {i} hub tokio scheduler overview"
        pages.update({u: f"site {i} page {j} unique tokio detail" for j, u in enumerate(children)})
        links[hub] = children
    search = InMemorySearchProvider(
        pages=[
            SearchHit(
                url=hub,
                title="Tokio runtime internals scheduler",
                snippet="tokio runtime internals scheduler tasks",
                published="2024-02-01",
                license_hint="MIT",
            )
            for hub in hubs
        ]
    )
    agent = SourceDiscoveryAgent(registry=InMemoryDatasetRegistry(), search=search)
    fetcher = InMemoryContentFetcher(pages=pages, links=links)
    channels = {SourceChannel.WEB: WebChannel(fetcher=fetcher)}

    result = run_engine_a(
        spec, work_dir=tmp_path, agent=agent, channels=channels, approver=AutoApprover()
    )

    assert result.metrics.pages_scraped <= 6
    sites = {c.source_url.split("/")[2] for c in result.chunks}
    assert sites == {"site0.dev", "site1.dev", "site2.dev"}


def test_engine_a_saturation_skips_remaining_web(minimal_spec_dict, tmp_path):
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Tokio runtime internals scheduler tasks."
    d["motivation"] = "teach tokio"
    spec = JobSpec.model_validate(d)

    # Five distinct URLs for the same aspect, all returning identical content.
    same = "tokio runtime internals scheduler tasks identical content block repeated"
    pages = {f"https://tokio.rs/p{i}": same for i in range(5)}
    search = InMemorySearchProvider(
        pages=[
            SearchHit(
                url=u,
                title="Tokio runtime internals scheduler",
                snippet="tokio runtime internals scheduler tasks",
                published="2024-02-01",
                license_hint="MIT",
            )
            for u in pages
        ]
    )
    agent = SourceDiscoveryAgent(registry=InMemoryDatasetRegistry(), search=search)
    channels = {SourceChannel.WEB: WebChannel(fetcher=InMemoryContentFetcher(pages=pages))}
    result = run_engine_a(
        spec, work_dir=tmp_path, agent=agent, channels=channels, approver=AutoApprover()
    )
    # identical content is also content-addressed to one blob, but saturation should
    # have stopped fetching after warm-up; assert a saturation note fired.
    assert any("saturated" in n for n in result.notes)
