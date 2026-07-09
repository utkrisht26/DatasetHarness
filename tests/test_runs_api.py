"""End-to-end tests for the runs API: discover -> approve -> run, fully offline.

The gateway app is built over an injected RunsService (offline discovery agent +
in-memory channels from test_orchestration) and a fake-crawler crawl service.
Covers the state machine (202/404/409/422), autonomous runs, restart persistence,
and metering.
"""

from __future__ import annotations

import copy
import time

from fastapi.testclient import TestClient

from adh.channels.web import Crawl4AIContentFetcher
from adh.channels.web.app import CrawlServiceApp
from adh.llm import TokenUsage, UsageTracker
from adh.service.app import create_app
from adh.service.routers.runs import RunsService

from .test_crawl_app import FakeCrawler
from .test_orchestration import _mixed_agent, _mixed_channels

TERMINAL = ("succeeded", "failed", "cancelled")


def _spec_body(minimal_spec_dict, **overrides) -> dict:
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns. Tokio runtime internals scheduler."
    d["motivation"] = "teach async"
    d.update(overrides)
    return d


def _make_client(tmp_path, agent_factory=None, channels=None) -> TestClient:
    runs = RunsService(
        tmp_path / "runs",
        agent_factory=agent_factory or (lambda usage: _mixed_agent()),
        channels=_mixed_channels() if channels is None else channels,
    )
    crawl = CrawlServiceApp(
        Crawl4AIContentFetcher(
            crawler_factory=lambda: FakeCrawler({}), run_config_factory=lambda req: None
        )
    )
    return TestClient(create_app(crawl_service=crawl, runs_service=runs))


def _await_status(client: TestClient, run_id: str, *statuses: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    run = None
    while time.monotonic() < deadline:
        run = client.get(f"/api/runs/{run_id}").json()
        if run["status"] in statuses:
            return run
        time.sleep(0.02)
    raise AssertionError(f"run {run_id} never reached {statuses}: {run}")


# --------------------------------------------------------------- happy path
def test_post_run_pauses_at_awaiting_approval(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        r = client.post("/api/runs", json=_spec_body(minimal_spec_dict))
        assert r.status_code == 202
        run_id = r.json()["run_id"]

        run = _await_status(client, run_id, "awaiting_approval")
        # the persisted plan is visible while parked, with discovery metrics
        assert run["plan"] is not None
        assert run["plan"]["sources"]
        assert run["metrics"]["cost_usd"] is None
        # and it appears in the list view
        listed = client.get("/api/runs").json()
        assert [x["run_id"] for x in listed] == [run_id]


def test_approve_subset_runs_to_succeeded(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()["run_id"]
        run = _await_status(client, run_id, "awaiting_approval")
        web_ids = [
            s["source_id"] for s in run["plan"]["sources"] if s["channel"] == "web"
        ]
        assert web_ids

        r = client.post(
            f"/api/runs/{run_id}/approve",
            json={"approved_source_ids": web_ids, "approve_synthesis": False},
        )
        assert r.status_code == 200
        final = _await_status(client, run_id, *TERMINAL)
        assert final["status"] == "succeeded"
        assert final["exit_reason"] == "completed"
        # the approval note landed in the run's durable notes
        assert any("approval:" in n for n in final["notes"])


def test_autonomous_run_skips_the_pause(minimal_spec_dict, tmp_path):
    body = _spec_body(minimal_spec_dict, discovery={"approval_gate": False})
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=body).json()["run_id"]
        final = _await_status(client, run_id, *TERMINAL)
        assert final["status"] == "succeeded"  # no approve call ever made


# --------------------------------------------------------------- error paths
def test_unknown_run_is_404_everywhere(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        assert client.get("/api/runs/nope").status_code == 404
        assert client.post("/api/runs/nope/approve", json={}).status_code == 404
        assert client.post("/api/runs/nope/cancel").status_code == 404


def test_double_approve_is_409(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()["run_id"]
        _await_status(client, run_id, "awaiting_approval")
        assert client.post(f"/api/runs/{run_id}/approve", json={}).status_code == 200
        second = client.post(f"/api/runs/{run_id}/approve", json={})
        assert second.status_code == 409
        assert "not awaiting approval" in second.json()["detail"]


def test_approve_with_unknown_source_ids_is_422(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()["run_id"]
        _await_status(client, run_id, "awaiting_approval")
        r = client.post(
            f"/api/runs/{run_id}/approve", json={"approved_source_ids": ["made:up:id"]}
        )
        assert r.status_code == 422
        assert "made:up:id" in r.json()["detail"]


def test_invalid_spec_body_is_422(tmp_path):
    with _make_client(tmp_path) as client:
        r = client.post("/api/runs", json={"name": "incomplete"})
        assert r.status_code == 422


def test_cancel_while_awaiting_then_approve_is_409(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()["run_id"]
        _await_status(client, run_id, "awaiting_approval")
        assert client.post(f"/api/runs/{run_id}/cancel").status_code == 200
        assert client.get(f"/api/runs/{run_id}").json()["status"] == "cancelled"
        assert client.post(f"/api/runs/{run_id}/approve", json={}).status_code == 409
        # terminal runs cannot be cancelled again
        assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409


# --------------------------------------------------------------- persistence
def test_awaiting_approval_survives_restart(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()["run_id"]
        _await_status(client, run_id, "awaiting_approval")

    # "restart": a brand-new service + app over the same base dir
    with _make_client(tmp_path) as client:
        run = client.get(f"/api/runs/{run_id}").json()
        assert run["status"] == "awaiting_approval"
        assert run["plan"]["sources"]  # plan.json reloaded from disk
        assert client.post(f"/api/runs/{run_id}/approve", json={}).status_code == 200
        final = _await_status(client, run_id, *TERMINAL)
        assert final["status"] == "succeeded"


# --------------------------------------------------------------- metering
def test_discovery_tokens_metered_through_the_api(minimal_spec_dict, tmp_path):
    def counting_agent(usage: UsageTracker):
        usage.add(TokenUsage(input_tokens=1000, output_tokens=250))  # as a model would
        return _mixed_agent()

    with _make_client(tmp_path, agent_factory=counting_agent) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()["run_id"]
        run = _await_status(client, run_id, "awaiting_approval")
        assert run["metrics"]["llm_tokens"] == 1250
        assert run["metrics"]["cost_usd"] is None  # provider gave no cost -> unknown


def test_discovery_over_budget_fails_without_pausing(minimal_spec_dict, tmp_path):
    def expensive_agent(usage: UsageTracker):
        usage.add(TokenUsage(input_tokens=5_000, output_tokens=0))
        return _mixed_agent()

    body = _spec_body(minimal_spec_dict, limits={"max_llm_tokens": 100})
    with _make_client(tmp_path, agent_factory=expensive_agent) as client:
        run_id = client.post("/api/runs", json=body).json()["run_id"]
        final = _await_status(client, run_id, *TERMINAL)
        assert final["status"] == "failed"
        assert final["exit_reason"] == "max_llm_tokens"
        # over-budget discovery never parks at the gate
        assert client.post(f"/api/runs/{run_id}/approve", json={}).status_code == 409


# --------------------------------------------------------------- MVP loop
def test_deep_agent_seeds_drive_the_crawler_end_to_end(minimal_spec_dict, tmp_path):
    """The core MVP: no user seeds, no search results — the deep agent proposes
    the starting links, and after approval the crawler starts exactly there."""
    from adh.channels import InMemoryContentFetcher, WebChannel
    from adh.contract import SourceChannel
    from adh.discovery import DeepAgentDiscoveryAgent
    from adh.search import InMemoryDatasetRegistry, InMemorySearchProvider

    from .test_deep_agent import ScriptedChatModel, _tool_call

    seed_url = "https://docs.rs/tokio/latest/tokio/"
    report = {
        "aspects": [
            {"topic": "rust async macros", "rationale": "core", "keywords": ["rust", "async"]}
        ],
        "sources": [
            {
                "channel": "web",
                "locator": seed_url,
                "title": "Tokio documentation index",
                "aspect_index": 0,
                "snippet": "tokio async runtime macros",
                "discovered_via": "model-knowledge",
                "license": "MIT",
                "shape_match": 0.7,
                "quality": 0.6,
                "est_yield": 30,
            }
        ],
        "synthesis_aspect_indexes": [],
        "notes": [],
    }

    def seed_proposing_agent(usage):
        return DeepAgentDiscoveryAgent(
            model=ScriptedChatModel(
                messages=iter([_tool_call("DiscoveryReport", report, "t1")])
            ),
            search=InMemorySearchProvider(),  # nothing to search: agent must propose
            registry=InMemoryDatasetRegistry(),
            usage=usage,
        )

    channels = {
        SourceChannel.WEB: WebChannel(
            fetcher=InMemoryContentFetcher(
                pages={seed_url: "# Tokio\nasync runtime macros: select, join, spawn"}
            )
        )
    }
    with _make_client(tmp_path, agent_factory=seed_proposing_agent, channels=channels) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()["run_id"]
        run = _await_status(client, run_id, "awaiting_approval")
        # the plan shows the agent-proposed seed at the approval gate
        assert [s["locator"] for s in run["plan"]["sources"]] == [seed_url]
        assert run["plan"]["sources"][0]["discovered_via"] == "model-knowledge"
        assert run["metrics"]["llm_tokens"] > 0  # the proposing call was metered

        client.post(f"/api/runs/{run_id}/approve", json={})
        final = _await_status(client, run_id, *TERMINAL)
        assert final["status"] == "succeeded"
        assert final["metrics"]["pages_scraped"] == 1  # crawled FROM the proposed seed


# --------------------------------------------------------------- gateway
def test_gateway_serves_crawl_under_its_prefix(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        assert client.get("/api/healthz").json()["status"] == "ok"
        assert client.get("/api/crawl/healthz").json()["status"] == "ok"
        assert client.get("/api/crawl/jobs").json() == []
        assert "llm" in client.get("/api/crawl/usage").json()
