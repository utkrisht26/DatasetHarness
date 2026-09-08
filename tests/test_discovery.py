"""Tests for the Source Discovery Agent + approval gate (Phase 1.0)."""

from __future__ import annotations

import copy
from pathlib import Path

from adh.contract import JobSpec, SourceChannel
from adh.discovery import (
    ApprovalDecision,
    AutoApprover,
    CallbackApprover,
    SourceDiscoveryAgent,
    ThresholdApprover,
    apply_decision,
    render_plan,
)
from adh.discovery.models import FetchPlan
from adh.search import (
    DatasetHit,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
    SearchHit,
)

DUMMY_PATH = Path(".")


# ----------------------------------------------------------------- helpers
def make_spec(minimal_spec_dict: dict, **discovery_over) -> JobSpec:
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Q/A pairs on advanced Rust async macro design (declarative + procedural)"
    d["motivation"] = "SFT data to teach a model idiomatic async macro patterns"
    d.setdefault("discovery", {})
    d["discovery"].update(discovery_over)
    return JobSpec.model_validate(d)


RUST_DATASET = DatasetHit(
    dataset_id="org/rust-async-macros-qa",
    locator="https://huggingface.co/datasets/org/rust-async-macros-qa",
    title="Rust async macro Q/A",
    description="rust async macro design declarative procedural question-answering input output",
    license="MIT",
    downloads=5000,
    last_modified="2024-09-01",
)

RUST_SAMPLE = (
    "Rust async macro examples with declarative and procedural macro design. "
    "question-answering input output pairs covering async macro patterns."
)


def registry_with_rust() -> InMemoryDatasetRegistry:
    return InMemoryDatasetRegistry(
        datasets=[RUST_DATASET],
        samples={RUST_DATASET.dataset_id: RUST_SAMPLE},
    )


def search_with_rust() -> InMemorySearchProvider:
    return InMemorySearchProvider(
        pages=[
            SearchHit(
                url="https://docs.rs/rust-async-macro-guide",
                title="Rust async macro design guide",
                snippet="declarative and procedural async macro design in rust with input output",
                published="2023-05-01",
                license_hint="Apache-2.0",
            ),
            SearchHit(
                url="https://reddit.com/r/rust/comments/xyz",
                title="rust async macro discussion",
                snippet="async macro rust declarative procedural",
                published="2024-01-01",
            ),
        ]
    )


# ----------------------------------------------------------------- decomposition
def test_decompose_produces_aspects(minimal_spec_dict):
    agent = SourceDiscoveryAgent()
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    assert len(plan.aspects) >= 1
    assert all(a.aspect_id for a in plan.aspects)
    # keywords should capture the domain
    kw = {k for a in plan.aspects for k in a.keywords}
    assert "rust" in kw and "macro" in kw


# ----------------------------------------------------------------- routing precedence
def test_reuse_first_prefers_datasets(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=registry_with_rust(), search=search_with_rust())
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    assert plan.sources, "expected at least one source"
    top = plan.sources[0]
    assert top.channel == SourceChannel.DATASETS
    assert top.candidate.locator == RUST_DATASET.locator


def test_web_fallback_when_no_dataset(minimal_spec_dict):
    agent = SourceDiscoveryAgent(
        registry=InMemoryDatasetRegistry(),  # empty -> no reuse
        search=search_with_rust(),
    )
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    assert plan.sources
    assert all(s.channel == SourceChannel.WEB for s in plan.sources)


def test_synthesis_when_no_external_source(minimal_spec_dict):
    agent = SourceDiscoveryAgent(
        registry=InMemoryDatasetRegistry(),
        search=InMemorySearchProvider(),  # empty web too
    )
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    assert plan.sources == ()
    # rust/async/macro are model-known -> synthesis aspects
    assert plan.synthesis_aspects
    assert any("rust" in a.keywords or "macro" in a.keywords for a in plan.synthesis_aspects)


def test_unknown_domain_no_synthesis_is_coverage_gap(minimal_spec_dict):
    spec_dict = copy.deepcopy(minimal_spec_dict)
    spec_dict["description"] = "obscure niche topic zzqq about widget flanges"
    spec_dict["motivation"] = "specialized flange knowledge"
    spec = JobSpec.model_validate(spec_dict)
    agent = SourceDiscoveryAgent(
        registry=InMemoryDatasetRegistry(), search=InMemorySearchProvider()
    )
    plan = agent.discover(spec, DUMMY_PATH)
    assert plan.is_empty
    assert any("coverage gap" in n for n in plan.notes)


# ----------------------------------------------------------------- filters
def test_exclude_domains_filters_web(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=InMemoryDatasetRegistry(), search=search_with_rust())
    spec = make_spec(minimal_spec_dict, constraints={"exclude_domains": ["reddit.com"]})
    plan = agent.discover(spec, DUMMY_PATH)
    locators = {s.candidate.locator for s in plan.sources}
    assert not any("reddit.com" in loc for loc in locators)


def test_include_domains_whitelist(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=InMemoryDatasetRegistry(), search=search_with_rust())
    spec = make_spec(minimal_spec_dict, constraints={"include_domains": ["docs.rs"]})
    plan = agent.discover(spec, DUMMY_PATH)
    assert plan.sources
    assert all("docs.rs" in s.candidate.locator for s in plan.sources)


def test_unknown_license_marked_not_ok(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=InMemoryDatasetRegistry(), search=search_with_rust())
    # reddit hit has no license_hint -> license_ok False
    spec = make_spec(minimal_spec_dict, constraints={"include_domains": ["reddit.com"]})
    plan = agent.discover(spec, DUMMY_PATH)
    assert plan.sources
    assert all(not s.evaluation.license_ok for s in plan.sources)


def test_freshness_penalizes_old(minimal_spec_dict):
    old_search = InMemorySearchProvider(
        pages=[
            SearchHit(
                url="https://docs.rs/old",
                title="old rust async macro",
                snippet="rust async macro declarative procedural input output",
                published="2019-01-01",
                license_hint="MIT",
            )
        ]
    )
    agent = SourceDiscoveryAgent(registry=InMemoryDatasetRegistry(), search=old_search)
    spec = make_spec(minimal_spec_dict, constraints={"recency_after": "2022-01-01"})
    plan = agent.discover(spec, DUMMY_PATH)
    assert plan.sources
    assert all(not s.evaluation.freshness_ok for s in plan.sources)


# ----------------------------------------------------------------- modes
def test_fixed_mode_wraps_seeds(minimal_spec_dict):
    agent = SourceDiscoveryAgent()
    spec = make_spec(
        minimal_spec_dict,
        mode="fixed",
        seeds=["https://docs.rs/x", "hf://org/ds"],
    )
    plan = agent.discover(spec, DUMMY_PATH)
    assert len(plan.sources) == 2
    channels = {s.channel for s in plan.sources}
    assert channels == {SourceChannel.WEB, SourceChannel.DATASETS}


def test_seeded_mode_enriches_goal(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=registry_with_rust(), search=search_with_rust())
    spec = make_spec(minimal_spec_dict, mode="seeded", seeds=["tokio internals"])
    plan = agent.discover(spec, DUMMY_PATH)
    assert plan.sources  # still discovers, seeds only enrich


# ----------------------------------------------------------------- re-discovery
def test_rediscover_merges_new_sources(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=InMemoryDatasetRegistry(), search=search_with_rust())
    spec = make_spec(minimal_spec_dict)
    plan = agent.discover(spec, DUMMY_PATH)
    before = len(plan.sources)
    # An aspect that the broader search can satisfy from a new registry hit.
    agent2 = SourceDiscoveryAgent(registry=registry_with_rust(), search=search_with_rust())
    aspect = plan.aspects[0]
    merged = agent2.rediscover(spec, aspect, existing=plan)
    assert len(merged.sources) >= before
    assert any("re-discovery" in n for n in merged.notes)


# ----------------------------------------------------------------- ranking
def test_sources_ranked_descending(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=registry_with_rust(), search=search_with_rust())
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    scores = [s.evaluation.score for s in plan.sources]
    assert scores == sorted(scores, reverse=True)


# ----------------------------------------------------------------- approval gate
def test_auto_approver_approves_all(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=registry_with_rust(), search=search_with_rust())
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    decision = AutoApprover().review(plan)
    approved = apply_decision(plan, decision)
    assert len(approved.sources) == len(plan.sources)


def test_threshold_approver_filters(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=registry_with_rust(), search=search_with_rust())
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    decision = ThresholdApprover(min_score=99.0).review(plan)  # impossible bar
    approved = apply_decision(plan, decision)
    assert approved.sources == ()


def test_callback_approver_picks_subset(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=registry_with_rust(), search=search_with_rust())
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    first_id = plan.sources[0].source_id

    def pick_first(p: FetchPlan) -> ApprovalDecision:
        return ApprovalDecision(approved_source_ids=frozenset({first_id}))

    decision = CallbackApprover(pick_first).review(plan)
    approved = apply_decision(plan, decision)
    assert len(approved.sources) == 1
    assert approved.sources[0].source_id == first_id


def test_apply_decision_can_drop_synthesis(minimal_spec_dict):
    agent = SourceDiscoveryAgent(
        registry=InMemoryDatasetRegistry(), search=InMemorySearchProvider()
    )
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    assert plan.synthesis_aspects
    decision = ApprovalDecision(approved_source_ids=frozenset(), approve_synthesis=False)
    approved = apply_decision(plan, decision)
    assert approved.synthesis_aspects == ()


def test_render_plan_is_readable(minimal_spec_dict):
    agent = SourceDiscoveryAgent(registry=registry_with_rust(), search=search_with_rust())
    plan = agent.discover(make_spec(minimal_spec_dict), DUMMY_PATH)
    text = render_plan(plan)
    assert "Fetch plan" in text
    assert "datasets" in text
