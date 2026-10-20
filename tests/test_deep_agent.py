"""Tests for the deepagents-backed discovery agent (step-2 plan, T4).

Fully offline: a scripted fake chat model drives the REAL ``create_deep_agent``
loop (tools, subagent wiring, ToolStrategy structured output) with no network and
no API key. The in-memory search/registry ports back the tools.
"""

from __future__ import annotations

import copy
import json
import logging

import pytest
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage

from adh.contract import JobSpec, SourceChannel
from adh.discovery import (
    DeepAgentDiscoveryAgent,
    DiscoveryAgentError,
    DiscoveryNoProgress,
    FallbackDiscoveryAgent,
    SourceDiscoveryAgent,
)
from adh.discovery.agent import DiscoveryConfig
from adh.discovery.deep_agent import (
    DiscoveryPlanningGate,
    DiscoveryReport,
    build_discovery_tools,
)
from adh.llm import UsageTracker
from adh.orchestration import plan_run
from adh.search import (
    DatasetHit,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
    SearchHit,
)
from adh.telemetry import read_telemetry


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

_PLAN_ARGS = {
    "aspects": ["rust async macros", "python basics"],
    "web_queries": ["rust async runtime internals"],
    "dataset_queries": ["rust async question answering dataset"],
    "rationale": "reuse a suitable dataset, then find authoritative web seeds",
}


def _plan_call(call_id: str = "plan", *, web_only: bool = False) -> AIMessage:
    args = copy.deepcopy(_PLAN_ARGS)
    if web_only:
        args["dataset_queries"] = []
    return _tool_call("submit_search_plan", args, call_id, in_tokens=0, out_tokens=0)


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
                _plan_call(),
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
    # metering: the plan, search, and report calls are folded; cost unknown -> None
    assert usage.calls == 3
    assert usage.total.total_tokens == 360
    assert usage.total.cost_usd is None


def test_recursion_limit_bounds_a_runaway_loop(minimal_spec_dict, tmp_path):
    """A model stuck re-calling tools hits the LangGraph recursion limit."""

    class LoopingModel(ScriptedChatModel):
        def __init__(self):
            def forever():
                yield _plan_call()
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


def test_structured_report_before_search_plan_is_rejected(minimal_spec_dict, tmp_path):
    model = ScriptedChatModel(messages=iter([_tool_call("DiscoveryReport", _REPORT_ARGS, "t1")]))
    with pytest.raises(DiscoveryAgentError, match="before submitting its search plan"):
        _agent(model).discover(_spec(minimal_spec_dict), tmp_path)


def test_report_embedded_in_topic_is_corrected_without_repeating_research(
    minimal_spec_dict, tmp_path
):
    """Valid JSON can still hide the whole report in a single string (bd39f7da)."""
    import httpx

    from adh.llm import build_chat_model

    malformed = {"aspects": [{"topic": "rust async macros',\n" + json.dumps(_REPORT_ARGS)}]}
    scripted = iter(
        [
            ("submit_search_plan", _PLAN_ARGS),
            ("web_search", {"query": "rust async runtime internals"}),
            ("read_file", {"file_path": "/research/web_search_1.json"}),
            ("DiscoveryReport", malformed),
            ("DiscoveryReport", _REPORT_ARGS),
        ]
    )
    requests = []

    def provider(request):
        requests.append(json.loads(request.content))
        name, args = next(scripted)
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{len(requests)}",
                "object": "chat.completion",
                "created": 0,
                "model": "diagnostic-model",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{len(requests)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(provider)) as client:
        model = build_chat_model(
            "openai:diagnostic-model",
            base_url="https://provider.test/v1",
            api_key="dummy",
            http_client=client,
        )
        usage = UsageTracker()
        planned = plan_run(
            _spec(minimal_spec_dict),
            work_dir=tmp_path,
            agent=_agent(model, usage=usage),
            usage=usage,
            run_id="report-correction",
        )

    assert len(planned.plan.sources) == 2
    assert [a.topic for a in planned.plan.aspects] == ["rust async macros", "python basics"]
    assert usage.calls == 5 and usage.total.total_tokens == 600
    assert len(list((tmp_path / "research").glob("web_search_*.json"))) == 1
    # The correction is in the same conversation, with both search and read results.
    tool_results = [m for m in requests[-1]["messages"] if m["role"] == "tool"]
    assert {m["tool_call_id"] for m in tool_results} == {"call-1", "call-2", "call-3", "call-4"}
    correction = next(m["content"] for m in tool_results if m["tool_call_id"] == "call-4")
    assert "sources: Field required" in correction
    assert "aspects.0.topic" in correction
    assert "without repeating completed searches" in correction
    events = read_telemetry(tmp_path)
    rejection = next(e for e in events if e["event"] == "report.rejected")
    assert rejection["attributes"]["action"] == "retry"
    assert rejection["attributes"]["tool_call_ids"] == ["call-4"]
    agent_done = next(e for e in events if e["event"] == "agent.completed")
    assert rejection["parent_operation_id"] == agent_done["operation_id"]
    assert agent_done["attributes"]["sources"] == len(_REPORT_ARGS["sources"])


@pytest.mark.parametrize("multiple", [False, True])
def test_invalid_reports_exhaust_bounded_corrections_and_fall_back(
    minimal_spec_dict, tmp_path, multiple
):
    def script():
        yield _plan_call()
        for i in range(3):
            yield AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "DiscoveryReport",
                        "args": _REPORT_ARGS if multiple else {},
                        "id": f"report-{i}-{j}",
                    }
                    for j in range(2 if multiple else 1)
                ],
            )
        pytest.fail("invalid report retries must stop before another model call")

    usage = UsageTracker()
    primary = _agent(ScriptedChatModel(messages=script()), usage=usage)
    planned = plan_run(
        _spec(minimal_spec_dict),
        work_dir=tmp_path,
        agent=FallbackDiscoveryAgent(primary=primary, fallback=SourceDiscoveryAgent()),
        usage=usage,
        run_id="invalid-report-fallback",
    )
    assert usage.calls == 4
    assert any("3 invalid DiscoveryReport" in note for note in planned.plan.notes)
    events = read_telemetry(tmp_path)
    rejected = [e["attributes"] for e in events if e["event"] == "report.rejected"]
    assert [e["action"] for e in rejected] == ["retry", "retry", "stop"]
    assert [e["retries_remaining"] for e in rejected] == [2, 1, 0]
    assert any(e["event"] == "agent.failed" for e in events)
    assert not any(e["event"] == "agent.completed" for e in events)


@pytest.mark.parametrize("synthesis", [False, True])
def test_explicit_empty_sources_remain_valid(minimal_spec_dict, tmp_path, synthesis):
    report = {
        "aspects": [{"topic": "python basics"}],
        "sources": [],
        "synthesis_aspect_indexes": [0] if synthesis else [],
        "notes": ["Stable topic for synthesis" if synthesis else "No suitable external sources"],
    }
    searches = (
        []
        if synthesis
        else [
            _tool_call("dataset_search", {"query": "nothing matches"}, "dataset"),
            _tool_call("web_search", {"query": "nothing matches"}, "web"),
        ]
    )
    model = ScriptedChatModel(
        messages=iter([_plan_call(), *searches, _tool_call("DiscoveryReport", report, "report")])
    )
    plan = _agent(model, search=InMemorySearchProvider()).discover(
        _spec(minimal_spec_dict), tmp_path
    )
    assert not plan.sources
    assert bool(plan.synthesis_aspects) == synthesis
    assert report["notes"][0] in plan.notes


def test_premature_empty_report_executes_plan_and_recovers(minimal_spec_dict, tmp_path):
    """995ed70e: blocked search, invalid plan, accepted plan, premature empty report."""
    seen_messages = []

    class ObservingModel(ScriptedChatModel):
        def _generate(self, messages, **kwargs):
            seen_messages.append(messages)
            return super()._generate(messages, **kwargs)

    empty = {"aspects": _REPORT_ARGS["aspects"], "sources": []}
    model = ObservingModel(
        messages=iter(
            [
                _tool_call("web_search", {"query": "unrelated NLRB work rules"}, "blocked"),
                _tool_call("submit_search_plan", {}, "invalid-plan"),
                _plan_call(web_only=True),
                _tool_call("DiscoveryReport", empty, "premature-report"),
                _tool_call("web_search", {"query": _PLAN_ARGS["web_queries"][0]}, "real-search"),
                _tool_call("read_file", {"file_path": "/research/web_search_1.json"}, "review"),
                _tool_call("DiscoveryReport", _REPORT_ARGS, "corrected-report"),
            ]
        )
    )
    usage = UsageTracker()
    planned = plan_run(
        _spec(minimal_spec_dict, discovery={"channels": ["web"]}),
        work_dir=tmp_path,
        agent=_agent(model, usage=usage),
        usage=usage,
        run_id="premature-empty-report",
    )
    assert [s.candidate.locator for s in planned.plan.sources] == ["https://tokio.rs/internals"]
    assert usage.calls == 7
    assert len(list((tmp_path / "research").glob("web_search_*.json"))) == 1
    feedback = next(
        m for m in seen_messages[4] if getattr(m, "tool_call_id", None) == "premature-report"
    )
    assert feedback.status == "error"
    assert 'web_search with {"query": "rust async runtime internals"}' in feedback.content
    assert "Dataset goal: Rust async macro patterns" in feedback.content
    assert "blocked/failed call does not count" in feedback.content
    events = read_telemetry(tmp_path)
    rejected = [e for e in events if e["event"] == "report.rejected"]
    assert len(rejected) == 1
    assert rejected[0]["attributes"]["kind"] == "completion"
    assert rejected[0]["attributes"]["tool_call_ids"] == ["premature-report"]
    assert next(e for e in events if e["event"] == "agent.completed")["attributes"]["sources"] > 0


@pytest.mark.parametrize(
    ("channels", "synthesis_indexes"),
    [
        (["web"], []),
        (["web"], [0, 1]),
        (["web"], [99]),
        (["web", "synthesis"], [0]),
    ],
)
def test_empty_report_cannot_skip_research_by_claiming_synthesis(
    minimal_spec_dict, tmp_path, channels, synthesis_indexes
):
    empty = {
        "aspects": _REPORT_ARGS["aspects"],
        "sources": [],
        "synthesis_aspect_indexes": synthesis_indexes,
        "notes": ["No suitable sources"],
    }
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(web_only=True),
                *[_tool_call("DiscoveryReport", empty, f"empty-{i}") for i in range(3)],
            ]
        )
    )
    usage = UsageTracker()

    class Search:
        def search(self, query, *, limit):
            return [
                SearchHit(
                    url="https://tokio.rs/internals",
                    title="Tokio",
                    snippet="rust async macro patterns",
                )
            ]

    primary = _agent(model, usage=usage, search=Search())
    planned = plan_run(
        _spec(minimal_spec_dict, discovery={"channels": channels}),
        work_dir=tmp_path,
        agent=FallbackDiscoveryAgent(
            primary=primary, fallback=SourceDiscoveryAgent(search=primary.search)
        ),
        usage=usage,
        run_id="empty-report-fallback",
    )
    assert usage.calls == 4
    assert planned.plan.sources or planned.plan.synthesis_aspects
    assert any("exhausted 2 correction retries" in note for note in planned.plan.notes)
    events = read_telemetry(tmp_path)
    rejected = [e for e in events if e["event"] == "report.rejected"]
    assert [e["attributes"]["action"] for e in rejected] == ["retry", "retry", "stop"]
    assert not any(e["event"] == "agent.completed" for e in events)


def test_empty_report_must_review_hits_and_explain_rejection(minimal_spec_dict, tmp_path):
    empty = {"aspects": _REPORT_ARGS["aspects"], "sources": []}
    explained = {**empty, "notes": ["Reviewed candidates do not match the requested record shape"]}
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(web_only=True),
                _tool_call("web_search", {"query": _PLAN_ARGS["web_queries"][0]}, "search"),
                _tool_call("DiscoveryReport", explained, "unread-report"),
                _tool_call("read_file", {"file_path": "/research/web_search_1.json"}, "review"),
                _tool_call("DiscoveryReport", empty, "unexplained-report"),
                _tool_call("DiscoveryReport", explained, "explained-report"),
            ]
        )
    )
    planned = plan_run(
        _spec(minimal_spec_dict, discovery={"channels": ["web"]}),
        work_dir=tmp_path,
        agent=_agent(model),
        run_id="reviewed-empty-report",
    )
    assert not planned.plan.sources
    assert explained["notes"][0] in planned.plan.notes
    rejected = [
        e["attributes"] for e in read_telemetry(tmp_path) if e["event"] == "report.rejected"
    ]
    assert len(rejected) == 2
    assert "results remain unreviewed" in rejected[0]["error"]
    assert "explain in notes" in rejected[1]["error"]


def test_synthesis_only_report_must_route_every_aspect(minimal_spec_dict, tmp_path):
    empty = {
        "aspects": _REPORT_ARGS["aspects"],
        "sources": [],
        "synthesis_aspect_indexes": [0],
        "notes": ["Stable synthesis topics"],
    }
    model = ScriptedChatModel(
        messages=iter(
            [
                _tool_call("submit_search_plan", {"aspects": ["rust", "python"]}, "plan"),
                _tool_call("DiscoveryReport", empty, "partial"),
                _tool_call(
                    "DiscoveryReport", {**empty, "synthesis_aspect_indexes": [0, 1]}, "complete"
                ),
            ]
        )
    )
    planned = plan_run(
        _spec(minimal_spec_dict, discovery={"channels": ["synthesis"]}),
        work_dir=tmp_path,
        agent=_agent(model),
        run_id="synthesis-only",
    )
    assert len(planned.plan.synthesis_aspects) == 2
    rejected = [e for e in read_telemetry(tmp_path) if e["event"] == "report.rejected"]
    assert len(rejected) == 1
    assert "every aspect" in rejected[0]["attributes"]["error"]


def test_schema_and_completion_failures_share_retry_limit(minimal_spec_dict, tmp_path):
    empty = {"aspects": _REPORT_ARGS["aspects"], "sources": []}
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(),
                _tool_call("DiscoveryReport", {}, "schema1"),
                _tool_call("DiscoveryReport", empty, "completion2"),
                _tool_call("DiscoveryReport", {}, "schema3"),
            ]
        )
    )
    with pytest.raises(DiscoveryAgentError, match="3 invalid DiscoveryReport"):
        plan_run(
            _spec(minimal_spec_dict), work_dir=tmp_path, agent=_agent(model), run_id="shared-limit"
        )
    rejected = [
        e["attributes"] for e in read_telemetry(tmp_path) if e["event"] == "report.rejected"
    ]
    assert [e["kind"] for e in rejected] == ["schema", "completion", "schema"]
    assert [e["action"] for e in rejected] == ["retry", "retry", "stop"]


@pytest.mark.parametrize(
    "report",
    [
        {"aspects": [{"topic": "rust async macros"}]},
        {"aspects": [], "sources": []},
        {"aspects": [{"topic": "   "}], "sources": []},
        {"aspects": [{"topic": "rust async macros", "sources": []}], "sources": []},
        {"aspects": [{"topic": "rust async macros"}], "sources": "[]"},
        {"aspects": [{"topic": json.dumps(_REPORT_ARGS)}], "sources": []},
    ],
)
def test_malformed_report_structure_is_not_silently_defaulted(report):
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        DiscoveryReport.model_validate(report)


def test_empty_completion_turn_is_retried_and_recovers(minimal_spec_dict, tmp_path):
    """A well-formed-but-empty completion (no text, no tool calls) is transient.

    Observed live: an OpenRouter upstream returned 200 with an entirely empty
    AIMessage (no usage either). The research invocation must be retried instead
    of killing deep discovery on the first hiccup.
    """
    model = ScriptedChatModel(
        messages=iter(
            [
                AIMessage(content=""),  # the empty completion
                _plan_call(),
                _tool_call("web_search", {"query": "rust async"}, "t1"),
                _tool_call("DiscoveryReport", _REPORT_ARGS, "t2", in_tokens=200, out_tokens=50),
            ]
        )
    )
    usage = UsageTracker()
    plan = _agent(model, usage=usage).discover(_spec(minimal_spec_dict), tmp_path)

    assert plan.aspects  # the retried invocation produced a real report
    # metering folds BOTH invocations — the wasted empty turn still counts
    assert usage.total.total_tokens == 360


def test_persistent_empty_completions_raise_specific_error(minimal_spec_dict, tmp_path):
    """Empty completions that survive the retries fail with an actionable reason."""
    model = ScriptedChatModel(
        messages=iter([AIMessage(content="") for _ in range(1 + 2)])  # initial + retries
    )
    with pytest.raises(DiscoveryAgentError) as excinfo:
        _agent(model).discover(_spec(minimal_spec_dict), tmp_path)
    assert "empty completion" in str(excinfo.value)
    assert "tool calling" in str(excinfo.value)  # points at the remedy: another model


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
        messages=iter([_plan_call(), _tool_call("DiscoveryReport", _REPORT_ARGS, "t1")])
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


def test_junk_sources_filtered_from_deep_agent_report(minimal_spec_dict, tmp_path):
    """Junk web sources reported by deep agent are filtered out with a note."""
    report = {
        "aspects": [
            {"topic": "rust async macros", "rationale": "core", "keywords": ["rust", "async"]}
        ],
        "sources": [
            {
                "channel": "web",
                "locator": "https://rust-lang.github.io/async-book",
                "title": "Async book",
                "aspect_index": 0,
                "shape_match": 0.8,
                "quality": 0.8,
                "est_yield": 10,
            },
            {
                "channel": "web",
                "locator": "https://www.indeed.com/q-async-programming-jobs.html",
                "title": "Now Hiring: 800 Async Programming Jobs | Indeed",
                "snippet": "now hiring async jobs salary search",
                "aspect_index": 0,
                "shape_match": 0.5,
                "quality": 0.5,
                "est_yield": 5,
            },
        ],
        "synthesis_aspect_indexes": [],
        "notes": [],
    }
    model = ScriptedChatModel(
        messages=iter([_plan_call(), _tool_call("DiscoveryReport", report, "t1")])
    )
    plan = _agent(model).discover(_spec(minimal_spec_dict), tmp_path)
    locators = [s.candidate.locator for s in plan.sources]
    assert "https://rust-lang.github.io/async-book" in locators
    assert "https://www.indeed.com/q-async-programming-jobs.html" not in locators
    assert any("junk source" in n for n in plan.notes)


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
    model = ScriptedChatModel(
        messages=iter([_plan_call(), _tool_call("DiscoveryReport", report, "t1")])
    )
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


def test_deep_report_caps_sources_per_aspect(minimal_spec_dict, tmp_path):
    report = {
        "aspects": [
            {
                "topic": "rust async macros",
                "rationale": "cover the target domain",
                "keywords": ["rust", "async", "macro"],
            }
        ],
        "sources": [
            {
                "aspect_index": 0,
                "channel": "web",
                "locator": f"https://docs.example.test/{index}",
                "title": f"Guide {index}",
                "snippet": "rust async macro guide",
                "license": "MIT",
                "published": "2025-01-01",
                "discovered_via": "web_search",
                "shape_match": 0.9,
                "quality": 0.8,
                "est_yield": 10 - index,
                "notes": "",
            }
            for index in range(5)
        ],
        "synthesis_aspect_indexes": [],
        "notes": [],
    }
    model = ScriptedChatModel(
        messages=iter([_plan_call(web_only=True), _tool_call("DiscoveryReport", report, "t1")])
    )
    plan = _agent(
        model,
        config=DiscoveryConfig(max_candidates_per_aspect=2),
    ).discover(_spec(minimal_spec_dict, discovery={"channels": ["web"]}), tmp_path)

    assert len(plan.sources) == 2
    assert any("maximum 2 per aspect" in note for note in plan.notes)


# ----------------------------------------------------------- observability
def test_budget_guard_stops_the_loop_before_more_spend(minimal_spec_dict, tmp_path, caplog):
    """Once discovery spend reaches the token budget the NEXT model turn is
    refused with DiscoveryBudgetExceeded (before the recursion limit matters)."""
    from adh.discovery.deep_agent import DiscoveryBudgetExceeded

    class LoopingModel(ScriptedChatModel):
        def __init__(self):
            def forever():
                yield _plan_call()
                i = 0
                while True:
                    i += 1
                    yield _tool_call("web_search", {"query": f"q{i}"}, f"t{i}")

            super().__init__(messages=forever())

    usage = UsageTracker()
    agent = _agent(LoopingModel(), usage=usage, token_budget=200)  # each turn costs 110
    with pytest.raises(DiscoveryBudgetExceeded):
        agent.discover(_spec(minimal_spec_dict), tmp_path)
    # spend is capped at roughly one turn past the budget, not 6 turns x 110
    assert usage.total.total_tokens < 400
    assert not any("callback" in record.getMessage() for record in caplog.records)
    assert not any(record.exc_info for record in caplog.records)


def test_budget_derives_from_spec_limits(minimal_spec_dict, tmp_path):
    """No explicit token_budget -> spec.limits.max_llm_tokens drives the guard."""
    from adh.discovery.deep_agent import DiscoveryBudgetExceeded

    class LoopingModel(ScriptedChatModel):
        def __init__(self):
            def forever():
                yield _plan_call()
                i = 0
                while True:
                    i += 1
                    yield _tool_call("web_search", {"query": f"q{i}"}, f"t{i}")

            super().__init__(messages=forever())

    usage = UsageTracker()
    agent = _agent(LoopingModel(), usage=usage)  # no explicit budget
    spec = _spec(minimal_spec_dict, limits={"max_llm_tokens": 150})
    with pytest.raises(DiscoveryBudgetExceeded):
        agent.discover(spec, tmp_path)


def test_no_progress_guard_stops_repeated_empty_samples(minimal_spec_dict, tmp_path, caplog):
    """Malformed sample retries stop after bounded evidence instead of burning the budget."""

    class EmptySamplingModel(ScriptedChatModel):
        def __init__(self):
            def forever():
                yield _plan_call()
                i = 0
                while True:
                    i += 1
                    yield _tool_call(
                        "sample_dataset",
                        {"dataset_id": f"}}invalid dataset {i}"},
                        f"t{i}",
                    )

            super().__init__(messages=forever())

    usage = UsageTracker()
    agent = _agent(
        EmptySamplingModel(),
        usage=usage,
        recursion_limit=100,
        max_consecutive_unproductive_tools=3,
    )
    with pytest.raises(DiscoveryNoProgress) as excinfo:
        agent.discover(_spec(minimal_spec_dict), tmp_path)

    assert "3 consecutive" in str(excinfo.value)
    assert "last tool: sample_dataset" in str(excinfo.value)
    assert usage.calls == 4
    assert not any("callback" in record.getMessage() for record in caplog.records)
    stopped = [record for record in caplog.records if "deep-agent stopped" in record.getMessage()]
    assert stopped and stopped[0].exc_info is None
    assert stopped[0].levelno == logging.WARNING


def test_repaired_search_can_continue_with_a_concrete_query(minimal_spec_dict, tmp_path):
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(),
                _tool_call("web_search", {"query": ":"}, "blocked1"),
                _tool_call("web_search", {"query": ""}, "blocked2"),
                _tool_call("web_search", {"query": "rust async runtime internals"}, "search"),
                _tool_call("sample_dataset", {"dataset_id": "org/missing"}, "sample1"),
                _tool_call("sample_dataset", {"dataset_id": "org/missing"}, "sample2"),
                _tool_call("DiscoveryReport", _REPORT_ARGS, "report"),
            ]
        )
    )
    plan = _agent(model).discover(_spec(minimal_spec_dict), tmp_path)
    assert "discovery: deepagents research loop" in plan.notes
    assert (tmp_path / "research" / "web_search_1.json").exists()
    assert (tmp_path / "research" / "web_search_2.json").exists()


def test_empty_search_file_reads_do_not_count_as_new_failed_searches(minimal_spec_dict, tmp_path):
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(),
                _tool_call("web_search", {"query": "no matches one"}, "search1"),
                _tool_call("read_file", {"file_path": "/research/web_search_1.json"}, "read1"),
                _tool_call("web_search", {"query": "no matches two"}, "search2"),
                _tool_call("read_file", {"file_path": "/research/web_search_2.json"}, "read2"),
                _tool_call("web_search", {"query": "no matches three"}, "search3"),
            ]
        )
    )
    usage = UsageTracker()
    with pytest.raises(DiscoveryNoProgress, match="3 consecutive"):
        _agent(model, usage=usage).discover(_spec(minimal_spec_dict), tmp_path)
    assert usage.calls == 6
    assert (tmp_path / "research" / "web_search_3.json").exists()


def test_parallel_search_hits_survive_empty_queries_and_duplicate_retry(
    minimal_spec_dict, tmp_path
):
    """Replay the bank-card failure, then let the agent review hits and finish."""
    import json

    queries = ["rust async", "runtime guide", "empty one", "empty two", "empty three", "empty four"]
    calls = []
    seen_messages = []

    class Search:
        def search(self, query, *, limit=8):
            calls.append(query)
            return (
                [SearchHit(url="https://tokio.rs/internals", title="Tokio", snippet="rust async")]
                if query in queries[:2]
                else []
            )

    class ObservingModel(ScriptedChatModel):
        def _generate(self, messages, **kwargs):
            seen_messages.append(messages)
            return super()._generate(messages, **kwargs)

    def script():
        yield _plan_call(web_only=True)
        yield AIMessage(
            content="",
            tool_calls=[
                {"name": "web_search", "args": {"query": query}, "id": f"s{i}"}
                for i, query in enumerate(queries)
            ],
        )
        files = {
            p: json.loads(p.read_text()) for p in (tmp_path / "research").glob("web_search_*.json")
        }
        empty = next(p for p, data in files.items() if not data["results"])
        yield _tool_call("read_file", {"file_path": f"/research/{empty.name}"}, "empty-read")
        yield _tool_call("web_search", {"query": files[empty]["query"]}, "repeat")
        for i, path in enumerate(p for p, data in files.items() if data["results"]):
            yield _tool_call("read_file", {"file_path": f"/research/{path.name}"}, f"review{i}")
        yield _tool_call("DiscoveryReport", _REPORT_ARGS, "report")

    planned = plan_run(
        _spec(minimal_spec_dict, discovery={"channels": ["web"]}),
        work_dir=tmp_path,
        run_id="parallel-hits",
        agent=_agent(ObservingModel(messages=script()), search=Search()),
    )
    assert "discovery: deepagents research loop" in planned.plan.notes
    assert len(calls) == 6  # the repeated empty query never reaches the provider
    assert len(list((tmp_path / "research").glob("web_search_*.json"))) == 6
    events = read_telemetry(tmp_path)
    assert not any(e["event"] == "guard.no_progress" for e in events)
    assert any("Search skipped:" in e.get("attributes", {}).get("output", "") for e in events)
    system = seen_messages[2][0].content
    assert "Unreviewed search results:" in str(system)
    assert "not relevance ranks" in str(system)


@pytest.mark.parametrize("review", [True, False])
def test_unread_hits_get_one_bounded_review_window(minimal_spec_dict, tmp_path, review):
    def script():
        yield _plan_call()
        yield _tool_call("web_search", {"query": "rust async runtime internals"}, "hit")
        for i in range(3):
            yield _tool_call("web_search", {"query": "nothing matches"}, f"empty{i}")
        if review:
            yield _tool_call("read_file", {"file_path": "/research/web_search_1.json"}, "review")
            yield _tool_call("DiscoveryReport", _REPORT_ARGS, "report")
        else:
            for i in range(3):
                yield _tool_call("web_search", {"query": "nothing matches"}, f"ignored{i}")

    agent = _agent(ScriptedChatModel(messages=script()))
    if review:
        planned = plan_run(
            _spec(minimal_spec_dict), work_dir=tmp_path, agent=agent, run_id="review"
        )
        assert "discovery: deepagents research loop" in planned.plan.notes
    else:
        with pytest.raises(DiscoveryNoProgress):
            plan_run(_spec(minimal_spec_dict), work_dir=tmp_path, agent=agent, run_id="ignored")
    events = read_telemetry(tmp_path)
    assert sum(e["event"] == "guard.review_pending" for e in events) == 1
    assert sum(e["event"] == "guard.no_progress" for e in events) == (0 if review else 1)


def test_schema_rejected_plans_still_have_a_bounded_retry_limit(minimal_spec_dict, tmp_path):
    model = ScriptedChatModel(
        messages=iter(
            [
                _tool_call(
                    "submit_search_plan", {"aspects": [], "web_queries": ["rust async"]}, f"bad{i}"
                )
                for i in range(3)
            ]
        )
    )
    usage = UsageTracker()
    with pytest.raises(DiscoveryNoProgress, match="3 consecutive"):
        _agent(model, usage=usage).discover(_spec(minimal_spec_dict), tmp_path)
    assert usage.calls == 3


def test_valid_results_containing_error_words_do_not_stop_discovery(minimal_spec_dict, tmp_path):
    query = "404 not found troubleshooting"
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(),
                *[_tool_call("web_search", {"query": query}, f"search{i}") for i in range(3)],
                _tool_call("DiscoveryReport", _REPORT_ARGS, "report"),
            ]
        )
    )
    search = InMemorySearchProvider(
        pages=[SearchHit(url="https://example.test/errors", title=query, snippet=query)]
    )
    plan = _agent(model, search=search).discover(_spec(minimal_spec_dict), tmp_path)
    assert "discovery: deepagents research loop" in plan.notes


@pytest.mark.parametrize("subagent_type", ["web-scout", "general-purpose"])
def test_no_progress_guard_stops_delegated_research(
    subagent_type, minimal_spec_dict, tmp_path, caplog
):
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(),
                _tool_call(
                    "task",
                    {
                        "subagent_type": subagent_type,
                        "description": "Execute the approved web plan",
                    },
                    "delegate",
                ),
                *[_tool_call("web_search", {"query": ":"}, f"blocked{i}") for i in range(3)],
            ]
        )
    )
    usage = UsageTracker()
    with pytest.raises(DiscoveryNoProgress, match="3 consecutive"):
        _agent(model, usage=usage, search=InMemorySearchProvider()).discover(
            _spec(minimal_spec_dict), tmp_path
        )
    assert usage.calls == 5
    assert not any("callback" in record.getMessage() for record in caplog.records)


def test_tool_results_carry_budget_status(tmp_path):
    """Every tool result appends the running budget line so the model can wrap up."""
    calls = {"n": 0}

    def status() -> str:
        calls["n"] += 1
        return f"discovery token budget: {calls['n'] * 100}/1000 used"

    tools = build_discovery_tools(
        InMemorySearchProvider(
            pages=[SearchHit(url="https://a.com", title="A", snippet="rust async")]
        ),
        InMemoryDatasetRegistry(),
        tmp_path,
        budget_status=status,
    )
    out = tools[0]("rust async")
    assert "[budget] discovery token budget: 100/1000 used" in out


def test_planning_gate_blocks_search_and_omits_disabled_channel_tools(tmp_path):
    gate = DiscoveryPlanningGate(frozenset({SourceChannel.WEB}))
    tools = build_discovery_tools(
        InMemorySearchProvider(
            pages=[SearchHit(url="https://a.com", title="A", snippet="rust async")]
        ),
        InMemoryDatasetRegistry(),
        tmp_path,
        allowed_channels={SourceChannel.WEB},
        planning_gate=gate,
    )
    by_name = {tool.__name__: tool for tool in tools}

    assert set(by_name) == {"submit_search_plan", "web_search"}
    assert "Search blocked" in by_name["web_search"]("rust async")
    assert not (tmp_path / "research" / "web_search_1.json").exists()

    placeholder = by_name["submit_search_plan"](["rust async"], [": "], [])
    assert "concrete, non-empty queries" in placeholder
    assert not gate.submitted

    rejected = by_name["submit_search_plan"](
        ["rust async"], ["rust async internals"], ["forbidden dataset query"]
    )
    assert "disabled channel" in rejected
    assert not gate.submitted

    accepted = by_name["submit_search_plan"](
        ["rust async"], ["rust async internals"], [], "find authoritative docs"
    )
    assert "Search plan accepted" in accepted
    assert gate.submitted
    assert (tmp_path / "research" / "search_plan.json").exists()
    assert "found 1 results" in by_name["web_search"]("rust async")


@pytest.mark.parametrize(
    ("tool_name", "expected_query"),
    [
        ("web_search", _PLAN_ARGS["web_queries"][0]),
        ("dataset_search", _PLAN_ARGS["dataset_queries"][0]),
    ],
)
@pytest.mark.parametrize("query", ["", ":", "   ", "…"])
def test_placeholder_search_executes_accepted_channel_query(
    tool_name, expected_query, query, tmp_path
):
    gate = DiscoveryPlanningGate(frozenset({SourceChannel.WEB, SourceChannel.DATASETS}))
    tools = build_discovery_tools(
        InMemorySearchProvider(),
        InMemoryDatasetRegistry(),
        tmp_path,
        planning_gate=gate,
        budget_status=lambda: "discovery token budget: 100/1000 used",
    )
    by_name = {tool.__name__: tool for tool in tools}
    by_name["submit_search_plan"](**_PLAN_ARGS)
    result = by_name[tool_name](query)
    assert f"Recovered invalid {tool_name} query" in result
    assert f"executed accepted-plan query {expected_query!r} instead" in result
    assert "[budget] discovery token budget: 100/1000 used" in result
    artifact = tmp_path / "research" / f"{tool_name}_1.json"
    assert artifact.exists()
    assert json.loads(artifact.read_text())["query"] == expected_query


def test_placeholder_query_uses_plan_and_deep_agent_recovers(minimal_spec_dict, tmp_path):
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(web_only=True),
                _tool_call("web_search", {"query": ": "}, "repaired-search"),
                _tool_call("read_file", {"file_path": "/research/web_search_1.json"}, "review"),
                _tool_call("DiscoveryReport", _REPORT_ARGS, "report"),
            ]
        )
    )
    usage = UsageTracker()
    planned = plan_run(
        _spec(minimal_spec_dict, discovery={"channels": ["web"]}),
        work_dir=tmp_path,
        agent=_agent(model, usage=usage),
        usage=usage,
        run_id="placeholder-recovery",
    )

    artifact = json.loads((tmp_path / "research" / "web_search_1.json").read_text())
    assert artifact["query"] == _PLAN_ARGS["web_queries"][0]
    assert [source.candidate.locator for source in planned.plan.sources] == [
        "https://tokio.rs/internals"
    ]
    assert usage.calls == 4
    assert not any("heuristic fallback" in note for note in planned.plan.notes)
    assert not any(event["event"] == "guard.no_progress" for event in read_telemetry(tmp_path))


def test_run_traces_steps_and_summarises_spend(minimal_spec_dict, caplog, tmp_path):
    """The loop logs every tool call, each model turn, and a finished summary."""
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(),
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


def test_run_persists_correlated_model_and_tool_events(minimal_spec_dict, tmp_path):
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(),
                _tool_call("web_search", {"query": "rust async"}, "t1"),
                _tool_call("DiscoveryReport", _REPORT_ARGS, "t2", in_tokens=200, out_tokens=50),
            ]
        )
    )
    usage = UsageTracker()
    plan_run(
        _spec(minimal_spec_dict),
        work_dir=tmp_path,
        agent=_agent(model, usage=usage),
        usage=usage,
        run_id="telemetry-test",
    )

    events = read_telemetry(tmp_path)
    names = [event["event"] for event in events]
    assert {"model.started", "model.completed", "tool.started", "tool.completed"} <= set(names)
    assert any(
        event["phase"] == "planning"
        and event["event"] == "tool.completed"
        and event["status"] == "succeeded"
        for event in events
    )
    assert (tmp_path / "research" / "search_plan.json").exists()
    starts = {
        event["operation_id"]
        for event in events
        if event["event"] in {"model.started", "tool.started"}
    }
    completions = {
        event["operation_id"]
        for event in events
        if event["event"] in {"model.completed", "tool.completed"}
    }
    assert starts <= completions
    assert any(
        event["event"] == "model.completed" and event["attributes"]["run_total_tokens"] == 360
        for event in events
    )


def test_discovery_automatically_captures_raw_and_parsed_http_responses(
    minimal_spec_dict, tmp_path
):
    import json

    import httpx

    from adh.llm import build_chat_model

    scripted = iter(
        [
            ("submit_search_plan", _PLAN_ARGS),
            ("web_search", {"query": ": "}),
            ("web_search", {"query": "rust async runtime internals"}),
            ("DiscoveryReport", _REPORT_ARGS),
        ]
    )

    def provider(request):
        name, args = next(scripted)
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{name}",
                "object": "chat.completion",
                "created": 0,
                "model": "diagnostic-model",
                "provider": "mock-provider",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{name}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(args)},
                                }
                            ],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
            },
        )

    with httpx.Client(transport=httpx.MockTransport(provider)) as client:
        model = build_chat_model(
            "openai:diagnostic-model",
            base_url="https://provider.test/v1",
            api_key="dummy",
            http_client=client,
        )
        usage = UsageTracker()
        plan_run(
            _spec(minimal_spec_dict),
            work_dir=tmp_path,
            agent=_agent(model, usage=usage),
            usage=usage,
            run_id="raw-response-test",
        )

    events = read_telemetry(tmp_path)
    starts = {e["operation_id"] for e in events if e["event"] == "model.started"}
    raw = [e for e in events if e["event"] == "model.response.raw"]
    parsed = [e for e in events if e["event"] == "model.response.parsed"]
    assert len(raw) == len(parsed) == 4
    assert starts == {e["operation_id"] for e in raw} == {e["operation_id"] for e in parsed}
    assert json.loads(
        raw[1]["attributes"]["response"]["body"]["choices"][0]["message"]["tool_calls"][0][
            "function"
        ]["arguments"]
    ) == {"query": ": "}
    assert parsed[1]["attributes"]["response"]["generations"][0][0]["message"]["tool_calls"][0][
        "args"
    ] == {"query": ": "}
    assert usage.calls == 4


def test_failure_logs_traceback_and_spend_then_reraises(minimal_spec_dict, caplog, tmp_path):
    """A runaway loop that hits the cap logs a traceback + spend, then propagates."""

    class LoopingModel(ScriptedChatModel):
        def __init__(self):
            def forever():
                yield _plan_call()
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
def test_no_progress_fallback_records_cause_without_traceback(minimal_spec_dict, tmp_path, caplog):
    model = ScriptedChatModel(
        messages=iter(
            [
                _plan_call(web_only=True),
                *[_tool_call("web_search", {"query": ":"}, f"blocked{i}") for i in range(3)],
            ]
        )
    )
    usage = UsageTracker()
    primary = _agent(model, usage=usage, search=InMemorySearchProvider())
    agent = FallbackDiscoveryAgent(
        primary=primary, fallback=SourceDiscoveryAgent(search=primary.search)
    )
    planned = plan_run(
        _spec(minimal_spec_dict, discovery={"channels": ["web"]}),
        work_dir=tmp_path,
        agent=agent,
        usage=usage,
        run_id="no-progress-fallback",
    )
    assert any("heuristic fallback" in note for note in planned.plan.notes)
    assert usage.calls == 4
    guards = [event for event in read_telemetry(tmp_path) if event["event"] == "guard.no_progress"]
    assert len(guards) == 1
    assert guards[0]["attributes"]["tool"] == "web_search"
    assert guards[0]["attributes"]["consecutive_unproductive"] == 3
    assert "Search blocked:" in guards[0]["attributes"]["output"]
    assert not any("callback" in record.getMessage() for record in caplog.records)
    assert not any(record.exc_info for record in caplog.records)
    assert not any(record.levelno >= logging.ERROR for record in caplog.records)


def test_fallback_agent_degrades_to_heuristic(minimal_spec_dict, tmp_path):
    """Agent error / no key -> heuristic plan, flagged in the notes."""

    class Exploding:
        def discover(self, spec, work_dir):
            raise DiscoveryAgentError("no api key")

    spec = _spec(minimal_spec_dict, discovery={"channels": ["web"]})
    fallback = FallbackDiscoveryAgent(primary=Exploding(), fallback=SourceDiscoveryAgent())
    planned = plan_run(spec, work_dir=tmp_path, agent=fallback, run_id="fallback-plan-test")
    plan = planned.plan
    assert plan.aspects  # the heuristic still decomposed the goal
    assert any("heuristic fallback" in n for n in plan.notes)

    import json

    search_plan = json.loads((tmp_path / "research" / "search_plan.json").read_text())
    assert search_plan["agent"] == "heuristic-agent"
    assert search_plan["allowed_channels"] == ["web"]
    assert search_plan["queries"]["datasets"] == []
    assert any(
        event["phase"] == "planning" and event["event"] == "search_plan.completed"
        for event in read_telemetry(tmp_path)
    )


def test_fallback_agent_passes_through_primary_success(minimal_spec_dict, tmp_path):
    model = ScriptedChatModel(
        messages=iter([_plan_call(), _tool_call("DiscoveryReport", _REPORT_ARGS, "t1")])
    )
    fallback = FallbackDiscoveryAgent(primary=_agent(model))
    plan = fallback.discover(_spec(minimal_spec_dict), tmp_path)
    assert "discovery: deepagents research loop" in plan.notes
    assert not any("heuristic fallback" in n for n in plan.notes)
