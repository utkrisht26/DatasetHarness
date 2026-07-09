"""Tests for the deepagents-backed discovery agent (step-2 plan, T4).

Fully offline: a scripted fake chat model drives the REAL ``create_deep_agent``
loop (tools, subagent wiring, ToolStrategy structured output) with no network and
no API key. The in-memory search/registry ports back the tools.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from adh.contract import JobSpec
from adh.discovery import (
    DeepAgentDiscoveryAgent,
    DiscoveryAgentError,
    FallbackDiscoveryAgent,
    SourceDiscoveryAgent,
)
from adh.discovery.deep_agent import build_discovery_tools
from adh.llm import UsageTracker
from adh.search import (
    DatasetHit,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
    SearchHit,
)


class ScriptedChatModel(GenericFakeChatModel):
    """Replays a fixed message script; tolerates tool/config binding."""

    def bind_tools(self, tools, **kwargs):
        return self

    def bind(self, **kwargs):
        return self


def _tool_call(name: str, args: dict, call_id: str, *, in_tokens=100, out_tokens=10) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": call_id}],
        usage_metadata={
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "total_tokens": in_tokens + out_tokens,
        },
    )


_REPORT_ARGS = {
    "aspects": [
        {"topic": "rust async macros", "rationale": "core topic", "keywords": ["rust", "async"]},
        {"topic": "python basics", "rationale": "model-known", "keywords": ["python"]},
    ],
    "sources": [
        {
            "channel": "web",
            "locator": "https://tokio.rs/internals",
            "title": "Tokio internals",
            "aspect_index": 0,
            "license": "MIT",
            "shape_match": 0.8,
            "quality": 0.7,
            "est_yield": 40,
            "discovered_via": "rust async runtime",
        },
        {
            "channel": "datasets",
            "locator": "https://huggingface.co/datasets/org/rust-qa",
            "title": "Rust QA",
            "aspect_index": 0,
            "license": "unknown-license",
            "shape_match": 0.9,
            "quality": 0.8,
            "est_yield": 50,
        },
        # rejected mappings: bad channel, bad aspect index
        {"channel": "carrier-pigeon", "locator": "x://nope", "aspect_index": 0},
        {"channel": "web", "locator": "https://ok.com/x", "aspect_index": 99},
    ],
    "synthesis_aspect_indexes": [1],
    "notes": ["agent note"],
}


def _agent(model, **kwargs) -> DeepAgentDiscoveryAgent:
    defaults = dict(
        model=model,
        search=InMemorySearchProvider(
            pages=[
                SearchHit(
                    url="https://tokio.rs/internals",
                    title="Tokio internals",
                    snippet="tokio runtime internals scheduler",
                    license_hint="MIT",
                )
            ]
        ),
        registry=InMemoryDatasetRegistry(),
    )
    defaults.update(kwargs)
    return DeepAgentDiscoveryAgent(**defaults)


def _spec(minimal_spec_dict, **overrides) -> JobSpec:
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns"
    d["motivation"] = "teach async"
    d.update(overrides)
    return JobSpec.model_validate(d)


# ----------------------------------------------------------- the research loop
def test_fake_model_tool_loop_yields_fetch_plan(minimal_spec_dict, tmp_path):
    """A scripted model runs the real create_deep_agent loop into a FetchPlan."""
    model = ScriptedChatModel(
        messages=iter(
            [
                _tool_call("web_search", {"query": "rust async"}, "t1"),
                _tool_call("DiscoveryReport", _REPORT_ARGS, "t2", in_tokens=200, out_tokens=50),
            ]
        )
    )
    usage = UsageTracker()
    plan = _agent(model, usage=usage).discover(_spec(minimal_spec_dict), tmp_path)

    assert [a.topic for a in plan.aspects] == ["rust async macros", "python basics"]
    # synthesis routed by index
    assert [a.topic for a in plan.synthesis_aspects] == ["python basics"]
    # two valid sources ranked; license verdict is harness-side, not the model's
    by_locator = {s.candidate.locator: s for s in plan.sources}
    assert set(by_locator) == {
        "https://tokio.rs/internals",
        "https://huggingface.co/datasets/org/rust-qa",
    }
    assert by_locator["https://tokio.rs/internals"].evaluation.license_ok  # MIT allowed
    assert not by_locator[
        "https://huggingface.co/datasets/org/rust-qa"
    ].evaluation.license_ok  # unknown license -> quarantined by the harness
    # rejected mappings surfaced as notes, not silently dropped
    assert any("unknown channel" in n for n in plan.notes)
    assert any("bad aspect index" in n for n in plan.notes)
    # metering: both calls' usage_metadata folded; cost unknown -> None
    assert usage.calls == 2
    assert usage.total.total_tokens == 360
    assert usage.total.cost_usd is None


def test_recursion_limit_bounds_a_runaway_loop(minimal_spec_dict, tmp_path):
    """A model stuck re-calling tools hits the LangGraph recursion limit."""

    class LoopingModel(ScriptedChatModel):
        def __init__(self):
            def forever():
                i = 0
                while True:
                    i += 1
                    yield _tool_call("web_search", {"query": f"q{i}"}, f"t{i}")

            super().__init__(messages=forever())

    usage = UsageTracker()
    agent = _agent(LoopingModel(), usage=usage, recursion_limit=6)
    with pytest.raises(Exception) as excinfo:
        agent.discover(_spec(minimal_spec_dict), tmp_path)
    assert "recursion" in str(excinfo.value).lower()
    assert usage.total.total_tokens > 0  # spend before the cap still counted


def test_junk_final_output_raises_discovery_agent_error(minimal_spec_dict, tmp_path):
    """A model that never emits the structured report is an agent error."""
    model = ScriptedChatModel(messages=iter([AIMessage(content="here is my plan, trust me")]))
    with pytest.raises(DiscoveryAgentError):
        _agent(model).discover(_spec(minimal_spec_dict), tmp_path)


def test_fixed_mode_skips_research_entirely(minimal_spec_dict, tmp_path):
    """Fixed mode wraps the user's seeds; the model is never invoked."""
    model = ScriptedChatModel(messages=iter([]))  # any call would StopIteration
    plan = _agent(model).discover(
        _spec(
            minimal_spec_dict,
            discovery={"mode": "fixed", "seeds": ["https://exact.com/page"]},
        ),
        tmp_path,
    )
    assert [s.candidate.locator for s in plan.sources] == ["https://exact.com/page"]


def test_domain_constraints_filter_web_sources(minimal_spec_dict, tmp_path):
    """Harness-side domain policy overrides whatever the model reported."""
    model = ScriptedChatModel(
        messages=iter([_tool_call("DiscoveryReport", _REPORT_ARGS, "t1")])
    )
    plan = _agent(model).discover(
        _spec(
            minimal_spec_dict,
            discovery={"constraints": {"exclude_domains": ["tokio.rs"]}},
        ),
        tmp_path,
    )
    assert "https://tokio.rs/internals" not in {s.candidate.locator for s in plan.sources}
    assert any("domain not allowed" in n for n in plan.notes)


def test_model_proposed_seeds_survive_with_empty_search_ports(minimal_spec_dict, tmp_path):
    """The core MVP: no user seeds + no search results -> the agent still hands
    the crawler starting links from its own knowledge (discovered_via=model-knowledge)."""
    report = {
        "aspects": [
            {"topic": "rust async macros", "rationale": "core", "keywords": ["rust", "async"]}
        ],
        "sources": [
            {
                "channel": "web",
                "locator": "https://docs.rs/tokio/latest/tokio/",
                "title": "Tokio documentation index",
                "aspect_index": 0,
                "snippet": "tokio async runtime macros select join spawn",
                "discovered_via": "model-knowledge",
                "shape_match": 0.6,
                "quality": 0.5,
                "est_yield": 30,
            }
        ],
        "synthesis_aspect_indexes": [],
        "notes": ["web_search returned nothing; proposed canonical docs as crawl seeds"],
    }
    model = ScriptedChatModel(messages=iter([_tool_call("DiscoveryReport", report, "t1")]))
    # BOTH ports empty: nothing to search, nothing to sample.
    agent = DeepAgentDiscoveryAgent(
        model=model, search=InMemorySearchProvider(), registry=InMemoryDatasetRegistry()
    )
    plan = agent.discover(_spec(minimal_spec_dict), tmp_path)

    assert len(plan.sources) == 1
    seed = plan.sources[0]
    assert seed.candidate.locator == "https://docs.rs/tokio/latest/tokio/"
    assert seed.candidate.discovered_via == "model-knowledge"
    # title + snippet carry the vocabulary the crawler steers its frontier with
    assert seed.candidate.title and seed.candidate.snippet


# ----------------------------------------------------------- observability
def test_run_traces_steps_and_summarises_spend(minimal_spec_dict, caplog, tmp_path):
    """The loop logs every tool call, each model turn, and a finished summary."""
    model = ScriptedChatModel(
        messages=iter(
            [
                _tool_call("web_search", {"query": "rust async"}, "t1"),
                _tool_call("DiscoveryReport", _REPORT_ARGS, "t2", in_tokens=200, out_tokens=50),
            ]
        )
    )
    with caplog.at_level(logging.INFO, logger="adh.discovery.deep_agent"):
        _agent(model, usage=UsageTracker()).discover(_spec(minimal_spec_dict), tmp_path)
    messages = [r.getMessage() for r in caplog.records]
    assert any("deep-agent starting" in m for m in messages)
    assert any("→ tool web_search" in m for m in messages)
    assert any("model turn" in m for m in messages)
    finished = [m for m in messages if "deep-agent finished" in m]
    assert finished and "360 token" in finished[0]  # spend folded into the summary


def test_failure_logs_traceback_and_spend_then_reraises(minimal_spec_dict, caplog, tmp_path):
    """A runaway loop that hits the cap logs a traceback + spend, then propagates."""

    class LoopingModel(ScriptedChatModel):
        def __init__(self):
            def forever():
                i = 0
                while True:
                    i += 1
                    yield _tool_call("web_search", {"query": f"q{i}"}, f"t{i}")

            super().__init__(messages=forever())

    agent = _agent(LoopingModel(), usage=UsageTracker(), recursion_limit=6)
    with caplog.at_level(logging.ERROR, logger="adh.discovery.deep_agent"):
        with pytest.raises(Exception):
            agent.discover(_spec(minimal_spec_dict), tmp_path)
    failed = [r for r in caplog.records if "deep-agent FAILED" in r.getMessage()]
    assert failed
    assert failed[0].exc_info is not None  # full traceback captured, not just a string
    assert "token" in failed[0].getMessage()


# ----------------------------------------------------------------- tools
def test_discovery_tools_wrap_the_ports(tmp_path):
    registry = InMemoryDatasetRegistry(
        datasets=[
            DatasetHit(
                dataset_id="org/rust-qa",
                locator="https://huggingface.co/datasets/org/rust-qa",
                title="Rust QA",
                description="rust async qa",
                license="MIT",
                downloads=10,
                last_modified="2024-01-01",
            )
        ],
        samples={"org/rust-qa": "Q: async? A: future"},
    )
    search = InMemorySearchProvider(
        pages=[SearchHit(url="https://a.com", title="A", snippet="rust async")]
    )
    web_search, dataset_search, sample_dataset = build_discovery_tools(search, registry, tmp_path)
    
    web_res = web_search("rust async")
    assert "saved to virtual path 'research/web_search_1.json'" in web_res
    web_file = tmp_path / "research" / "web_search_1.json"
    assert web_file.exists()
    import json
    web_data = json.loads(web_file.read_text(encoding="utf-8"))
    assert web_data["results"][0]["url"] == "https://a.com"

    ds_res = dataset_search("rust async")
    assert "saved to virtual path 'research/dataset_search_1.json'" in ds_res
    ds_file = tmp_path / "research" / "dataset_search_1.json"
    assert ds_file.exists()
    ds_data = json.loads(ds_file.read_text(encoding="utf-8"))
    assert ds_data["results"][0]["dataset_id"] == "org/rust-qa"

    sample_res = sample_dataset("org/rust-qa")
    assert "saved to virtual path 'research/dataset_sample_org_rust-qa.txt'" in sample_res
    sample_file = tmp_path / "research" / "dataset_sample_org_rust-qa.txt"
    assert sample_file.exists()
    assert sample_file.read_text(encoding="utf-8") == "Q: async? A: future"

    assert "No sample content available" in sample_dataset("org/missing")


# ----------------------------------------------------------------- fallback
def test_fallback_agent_degrades_to_heuristic(minimal_spec_dict, tmp_path):
    """Agent error / no key -> heuristic plan, flagged in the notes."""

    class Exploding:
        def discover(self, spec, work_dir):
            raise DiscoveryAgentError("no api key")

    spec = _spec(minimal_spec_dict)
    fallback = FallbackDiscoveryAgent(primary=Exploding(), fallback=SourceDiscoveryAgent())
    plan = fallback.discover(spec, tmp_path)
    assert plan.aspects  # the heuristic still decomposed the goal
    assert any("heuristic fallback" in n for n in plan.notes)


def test_fallback_agent_passes_through_primary_success(minimal_spec_dict, tmp_path):
    model = ScriptedChatModel(
        messages=iter([_tool_call("DiscoveryReport", _REPORT_ARGS, "t1")])
    )
    fallback = FallbackDiscoveryAgent(primary=_agent(model))
    plan = fallback.discover(_spec(minimal_spec_dict), tmp_path)
    assert "discovery: deepagents research loop" in plan.notes
    assert not any("heuristic fallback" in n for n in plan.notes)
