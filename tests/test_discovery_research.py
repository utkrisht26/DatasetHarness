"""Parallel progress, candidate review, and search retry behavior."""

from concurrent.futures import ThreadPoolExecutor
from itertools import permutations
from threading import Barrier

import pytest
from langchain_core.tools import StructuredTool
from pydantic import ValidationError

from adh.contract import SourceChannel
from adh.discovery.deep_agent import DiscoveryPlanningGate, build_discovery_tools
from adh.discovery.research import DiscoveryResearchState
from adh.search import InMemoryDatasetRegistry, InMemorySearchProvider


def _call(name, call_id, **args):
    return {"id": call_id, "name": name, "args": args}


def test_mixed_parallel_batch_is_productive_in_every_completion_order():
    calls = [_call("web_search", str(i)) for i in range(4)]
    for order in permutations(calls):
        state = DiscoveryResearchState()
        state.start_batch("mixed", calls)
        for i, call in enumerate(order):
            result = state.complete_tool(
                call, unproductive=call["id"] != "0", failed=False, limit=3
            )
            if i < 3:
                assert result is None  # even three empty siblings cannot abort the batch
        assert result.action == "continue"
        assert result.consecutive_unproductive == 0


def test_empty_parallel_batches_count_once_and_stop_after_limit():
    state = DiscoveryResearchState()
    for batch in range(3):
        calls = [_call("web_search", f"{batch}-{i}") for i in range(6)]
        state.start_batch(str(batch), calls)
        for call in calls:
            result = state.complete_tool(call, unproductive=True, failed=False, limit=3)
        assert result.consecutive_unproductive == batch + 1
        assert result.action == ("stop" if batch == 2 else "continue")


def test_only_new_successful_candidate_reads_reset_progress():
    state = DiscoveryResearchState()
    state.record_search("web_search", "hit", "research/hit.json", 8)
    state.record_search("web_search", "empty", "research/empty.json", 0)

    def finish(call, *, unproductive=False, failed=False):
        state.start_batch(call["id"], [call])
        return state.complete_tool(call, unproductive=unproductive, failed=failed, limit=3)

    assert finish(_call("web_search", "empty1"), unproductive=True).consecutive_unproductive == 1
    assert (
        finish(
            _call("read_file", "read-empty", file_path="/research/empty.json")
        ).consecutive_unproductive
        == 1
    )
    assert (
        finish(
            _call("read_file", "failed-read", file_path="/research/hit.json"), failed=True
        ).consecutive_unproductive
        == 1
    )
    assert "research/hit.json" in state.review_hint()
    assert "research/empty.json" not in state.review_hint()
    assert (
        finish(
            _call("read_file", "read-hit", file_path="/research/hit.json")
        ).consecutive_unproductive
        == 0
    )
    assert state.review_hint() == ""
    finish(_call("web_search", "empty2"), unproductive=True)
    assert (
        finish(
            _call("read_file", "reread", file_path="/research/hit.json")
        ).consecutive_unproductive
        == 1
    )


@pytest.mark.parametrize("tool_name", ["web_search", "dataset_search"])
def test_concurrent_duplicate_empty_queries_make_one_provider_request(tmp_path, tool_name):
    calls = []
    barrier = Barrier(2)

    class Provider:
        def search(self, query, *, limit):
            calls.append(query)
            return []

    provider = Provider()
    by_name = {t.__name__: t for t in build_discovery_tools(provider, provider, tmp_path)}

    def search():
        barrier.wait(timeout=5)
        return by_name[tool_name]("no results")

    with ThreadPoolExecutor(max_workers=2) as pool:
        outputs = list(pool.map(lambda _: search(), range(2)))
    assert calls == ["no results"]
    assert sum("Search skipped:" in output for output in outputs) == 1
    assert sum("skip reading" in output for output in outputs) == 1
    by_name[tool_name]("a broader query")
    assert calls == ["no results", "a broader query"]


def test_plan_schema_requires_at_least_one_aspect(tmp_path):
    gate = DiscoveryPlanningGate(frozenset({SourceChannel.WEB}))
    tools = build_discovery_tools(
        InMemorySearchProvider(),
        InMemoryDatasetRegistry(),
        tmp_path,
        planning_gate=gate,
        allowed_channels={SourceChannel.WEB},
    )
    submit = next(t for t in tools if t.__name__ == "submit_search_plan")
    tool = StructuredTool.from_function(submit)
    assert tool.args_schema.model_json_schema()["properties"]["aspects"]["minItems"] == 1
    with pytest.raises(ValidationError):
        tool.invoke({"aspects": [], "web_queries": ["rust async"]})
    assert "rejected" in submit(["   "], ["rust async"])
    assert not gate.submitted


def test_completion_evidence_requires_finished_provider_calls(tmp_path):
    state = DiscoveryResearchState()
    gate = DiscoveryPlanningGate(frozenset({SourceChannel.WEB, SourceChannel.DATASETS}))

    class FailingSearch:
        def search(self, query, *, limit):
            raise RuntimeError("provider unavailable")

    by_name = {
        tool.__name__: tool
        for tool in build_discovery_tools(
            FailingSearch(),
            InMemoryDatasetRegistry(),
            tmp_path,
            planning_gate=gate,
            research_state=state,
        )
    }
    required = {"web_search", "dataset_search"}
    assert "Search blocked" in by_name["web_search"]("rust async")
    assert state.missing_searches(required) == sorted(required)
    by_name["submit_search_plan"](["rust"], ["rust async"], ["rust dataset"])
    assert state.missing_searches(required) == sorted(required)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        by_name["web_search"]("rust async")
    assert state.missing_searches(required) == sorted(required)
    assert "found 0 results" in by_name["dataset_search"]("rust dataset")
    assert state.missing_searches(required) == ["web_search"]
