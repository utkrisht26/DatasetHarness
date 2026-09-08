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


def _make_client(
    tmp_path, agent_factory=None, channels=None, **runs_kwargs
) -> TestClient:
    runs = RunsService(
        tmp_path / "runs",
        agent_factory=agent_factory or (lambda usage: _mixed_agent()),
        channels=_mixed_channels() if channels is None else channels,
        **runs_kwargs,
    )
    crawl = CrawlServiceApp(
        Crawl4AIContentFetcher(
            crawler_factory=lambda: FakeCrawler({}), run_config_factory=lambda req: None
        )
    )
    return TestClient(create_app(crawl_service=crawl, runs_service=runs))


def _await_status(
    client: TestClient, run_id: str, *statuses: str, timeout: float = 10.0
) -> dict:
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
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
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
        assert client.post("/api/runs/nope/restart").status_code == 404
        assert client.get("/api/runs/nope/spec").status_code == 404


# --------------------------------------------------------------- restart
def test_restart_cancelled_run_starts_fresh_with_same_spec(minimal_spec_dict, tmp_path):
    """Restarting a dead run must not touch its audit trail: a NEW run is queued
    from the same saved spec, noted as a restart, and drives to the gate again."""
    import json as _json

    with _make_client(tmp_path) as client:
        original = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        _await_status(client, original, "awaiting_approval")
        client.post(f"/api/runs/{original}/cancel")

        r = client.post(f"/api/runs/{original}/restart")
        assert r.status_code == 202
        fresh = r.json()
        assert fresh["run_id"] != original
        assert fresh["status"] == "discovering"  # queued before the response

        # the cancelled run is untouched history
        assert client.get(f"/api/runs/{original}").json()["status"] == "cancelled"

        # identical config: the new dir persists the very same spec document
        old_spec = _json.loads((tmp_path / "runs" / original / "spec.json").read_text())
        new_spec = _json.loads(
            (tmp_path / "runs" / fresh["run_id"] / "spec.json").read_text()
        )
        assert old_spec == new_spec

        # offline world: the fresh run re-discovers and parks at the gate again,
        # carrying the restart provenance note
        parked = _await_status(client, fresh["run_id"], *TERMINAL, "awaiting_approval")
        assert parked["status"] == "awaiting_approval"
        assert any("restarted from" in n for n in parked["notes"])
        # and both runs show up in the list view
        listed_ids = {x["run_id"] for x in client.get("/api/runs").json()}
        assert listed_ids == {original, fresh["run_id"]}


def test_restart_failed_run_reruns_discovery(minimal_spec_dict, tmp_path):
    """A run whose discovery crashed lands FAILED (not stuck) and restarts clean."""
    calls = {"n": 0}

    def flaky_factory(usage):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("search provider down")
        return _mixed_agent()

    with _make_client(tmp_path, agent_factory=flaky_factory) as client:
        dead = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        final = _await_status(client, dead, *TERMINAL)
        assert final["status"] == "failed"
        assert "search provider down" in final["error"]

        r = client.post(f"/api/runs/{dead}/restart")
        assert r.status_code == 202
        parked = _await_status(
            client, r.json()["run_id"], *TERMINAL, "awaiting_approval"
        )
        assert parked["status"] == "awaiting_approval"  # healthy agent -> full rerun


def test_restart_non_terminal_run_is_409(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        _await_status(client, run_id, "awaiting_approval")
        r = client.post(f"/api/runs/{run_id}/restart")
        assert r.status_code == 409
        assert "awaiting_approval" in r.json()["detail"]
        assert "terminal" in r.json()["detail"]


def test_double_approve_is_409(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        _await_status(client, run_id, "awaiting_approval")
        assert client.post(f"/api/runs/{run_id}/approve", json={}).status_code == 200
        second = client.post(f"/api/runs/{run_id}/approve", json={})
        assert second.status_code == 409
        assert "not awaiting approval" in second.json()["detail"]


def test_approve_with_unknown_source_ids_is_422(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
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
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        _await_status(client, run_id, "awaiting_approval")
        assert client.post(f"/api/runs/{run_id}/cancel").status_code == 200
        assert client.get(f"/api/runs/{run_id}").json()["status"] == "cancelled"
        assert client.post(f"/api/runs/{run_id}/approve", json={}).status_code == 409
        # terminal runs cannot be cancelled again
        assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409


class _SlowChannel:
    """Wraps the offline channels; each fetch holds long enough to cancel mid-run."""

    def __init__(self, inner) -> None:
        self._inner = inner

    def fetch(self, source, ctx):
        time.sleep(0.5)
        return self._inner.fetch(source, ctx)

    def close(self) -> None:
        close = getattr(self._inner, "close", None)
        if callable(close):
            close()


def test_cancel_during_acquisition_stops_mid_run(minimal_spec_dict, tmp_path):
    """An operator can cancel an in-flight acquisition: 200 while ACQUIRING, and
    the worker honours it between sources — the run lands CANCELLED."""
    from adh.contract import SourceChannel

    slow = {
        SourceChannel.DATASETS: _SlowChannel(_mixed_channels()[SourceChannel.DATASETS]),
        SourceChannel.WEB: _SlowChannel(_mixed_channels()[SourceChannel.WEB]),
    }
    body = _spec_body(minimal_spec_dict, discovery={"approval_gate": False})
    with _make_client(tmp_path, channels=slow) as client:
        run_id = client.post("/api/runs", json=body).json()["run_id"]
        _await_status(client, run_id, "acquiring")

        r = client.post(f"/api/runs/{run_id}/cancel")
        assert r.status_code == 200
        assert r.json()["status"] == "cancelled"

        # cancel() itself makes the status terminal; the worker's acknowledgement
        # is exit_reason flipping from None to "cancelled" between sources
        deadline = time.monotonic() + 10
        final = client.get(f"/api/runs/{run_id}").json()
        while time.monotonic() < deadline and final["exit_reason"] != "cancelled":
            time.sleep(0.02)
            final = client.get(f"/api/runs/{run_id}").json()
        assert final["exit_reason"] == "cancelled"
        assert final["status"] == "cancelled"
        assert any("cancelled" in n for n in final["notes"])
        # a cancelled run cannot be built or re-cancelled
        assert client.post(f"/api/runs/{run_id}/build").status_code == 409
        assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409


# --------------------------------------------------------------- persistence
def test_awaiting_approval_survives_restart(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
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
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        run = _await_status(client, run_id, "awaiting_approval")
        assert run["metrics"]["llm_tokens"] == 1250
        assert run["metrics"]["cost_usd"] is None  # provider gave no cost -> unknown


def test_discovery_over_budget_parks_usable_plan_for_review(
    minimal_spec_dict, tmp_path
):
    def expensive_agent(usage: UsageTracker):
        usage.add(TokenUsage(input_tokens=5_000, output_tokens=0))
        return _mixed_agent()

    body = _spec_body(minimal_spec_dict, limits={"max_llm_tokens": 100})
    with _make_client(tmp_path, agent_factory=expensive_agent) as client:
        run_id = client.post("/api/runs", json=body).json()["run_id"]
        # paid-for discovery parks for a human decision instead of being discarded
        parked = _await_status(client, run_id, "awaiting_approval")
        assert parked["status"] == "awaiting_approval"
        # approving anyway holds the same ceiling during acquisition -> failed
        resp = client.post(f"/api/runs/{run_id}/approve", json={})
        assert resp.status_code == 200
        final = _await_status(client, run_id, *TERMINAL)
        assert final["status"] == "failed"
        assert final["exit_reason"] == "max_llm_tokens"


# --------------------------------------------------------------- Engine B build
def _offline_build_client(tmp_path, **overrides) -> TestClient:
    from adh.generation import HeuristicGenerator
    from adh.verification import HeuristicJudge

    kwargs = {
        "engine_b_generator": HeuristicGenerator(),
        "engine_b_judge": HeuristicJudge(),
    }
    kwargs.update(overrides)
    return _make_client(tmp_path, **kwargs)


def _run_to_succeeded(client: TestClient, minimal_spec_dict, **spec_overrides) -> str:
    body = _spec_body(minimal_spec_dict, **spec_overrides)
    run_id = client.post("/api/runs", json=body).json()["run_id"]
    _await_status(client, run_id, "awaiting_approval")
    client.post(f"/api/runs/{run_id}/approve", json={})
    _await_status(client, run_id, *TERMINAL)
    return run_id


def test_build_endpoint_produces_a_dataset(minimal_spec_dict, tmp_path):
    with _offline_build_client(tmp_path) as client:
        run_id = _run_to_succeeded(
            client, minimal_spec_dict, target_size={"records": 6}
        )

        # nothing built yet -> the dataset endpoint says so
        assert client.get(f"/api/runs/{run_id}/dataset").status_code == 404

        r = client.post(f"/api/runs/{run_id}/build")
        assert r.status_code == 202
        assert r.json()["status"] == "building"  # transitioned before the worker ran

        summary = None
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            resp = client.get(f"/api/runs/{run_id}/dataset")
            if resp.status_code == 200:
                summary = resp.json()
                break
            time.sleep(0.05)
        assert summary is not None
        assert summary["counts"]["total_records"] > 0

        final = _await_status(client, run_id, "succeeded")  # BUILDING is transient
        assert any("engine B build finished" in n for n in final["notes"])
        # a second build after completion is legal (rebuild), not a conflict
        rebuild = client.post(f"/api/runs/{run_id}/build")
        assert rebuild.status_code == 202


def test_double_build_in_flight_is_409(minimal_spec_dict, tmp_path):
    import time as _time

    from adh.generation import HeuristicGenerator

    class SlowGenerator(HeuristicGenerator):
        def generate(self, seed, *, spec, store=None):
            _time.sleep(0.5)
            return super().generate(seed, spec=spec, store=store)

    with _offline_build_client(tmp_path, engine_b_generator=SlowGenerator()) as client:
        run_id = _run_to_succeeded(
            client, minimal_spec_dict, target_size={"records": 6}
        )
        assert client.post(f"/api/runs/{run_id}/build").status_code == 202
        second = client.post(f"/api/runs/{run_id}/build")
        assert second.status_code == 409
        assert "building" in second.json()["detail"]
        _await_status(client, run_id, "succeeded")


def test_failed_build_returns_to_succeeded_with_note(minimal_spec_dict, tmp_path):
    # A generator error is absorbed per-seed inside Engine B; only a failure that
    # escapes run_engine_b (e.g. the jury) exercises the build-failure path.
    class ExplodingJudge:
        def evaluate(self, **kwargs):
            raise RuntimeError("jury exploded")

    from adh.generation import HeuristicGenerator

    with _offline_build_client(
        tmp_path,
        engine_b_generator=HeuristicGenerator(),
        engine_b_judge=ExplodingJudge(),
    ) as client:
        run_id = _run_to_succeeded(client, minimal_spec_dict)
        assert client.post(f"/api/runs/{run_id}/build").status_code == 202
        final = _await_status(client, run_id, "succeeded")
        assert any(
            "engine B build failed" in n and "jury exploded" in n
            for n in final["notes"]
        )
        assert client.get(f"/api/runs/{run_id}/dataset").status_code == 404


def test_build_on_unsucceeded_run_and_unknown_run_conflict(minimal_spec_dict, tmp_path):
    with _offline_build_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        _await_status(client, run_id, "awaiting_approval")
        r = client.post(f"/api/runs/{run_id}/build")
        assert r.status_code == 409
        assert "only a succeeded run" in r.json()["detail"]
        assert client.get("/api/runs/nope/dataset").status_code == 404
        assert client.post("/api/runs/nope/build").status_code == 404


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
            {
                "topic": "rust async macros",
                "rationale": "core",
                "keywords": ["rust", "async"],
            }
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
    with _make_client(
        tmp_path, agent_factory=seed_proposing_agent, channels=channels
    ) as client:
        run_id = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        run = _await_status(client, run_id, "awaiting_approval")
        # the plan shows the agent-proposed seed at the approval gate
        assert [s["locator"] for s in run["plan"]["sources"]] == [seed_url]
        assert run["plan"]["sources"][0]["discovered_via"] == "model-knowledge"
        assert run["metrics"]["llm_tokens"] > 0  # the proposing call was metered

        client.post(f"/api/runs/{run_id}/approve", json={})
        final = _await_status(client, run_id, *TERMINAL)
        assert final["status"] == "succeeded"
        assert final["metrics"]["pages_scraped"] == 1  # crawled FROM the proposed seed


# --------------------------------------------------------------- edit-restart
def test_get_run_spec_returns_the_saved_config(minimal_spec_dict, tmp_path):
    """The edit-restart flow prefills its editor from the run's saved spec."""
    body = _spec_body(minimal_spec_dict)
    with _make_client(tmp_path) as client:
        run_id = client.post("/api/runs", json=body).json()["run_id"]
        _await_status(client, run_id, "awaiting_approval")

        r = client.get(f"/api/runs/{run_id}/spec")
        assert r.status_code == 200
        saved = r.json()
        assert saved["name"] == body["name"]
        assert saved["description"] == body["description"]
        assert saved["motivation"] == body["motivation"]


def test_restart_with_edited_spec_runs_from_the_new_config(minimal_spec_dict, tmp_path):
    """Restart accepts an edited spec: the fresh run uses it; the original
    audit trail keeps the old config and the note says the config changed."""
    import json as _json

    with _make_client(tmp_path) as client:
        original = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        _await_status(client, original, "awaiting_approval")
        client.post(f"/api/runs/{original}/cancel")

        edited = _spec_body(minimal_spec_dict)
        edited["name"] = "edited-name"
        edited["motivation"] = "revised motivation"

        r = client.post(f"/api/runs/{original}/restart", json={"spec": edited})
        assert r.status_code == 202
        fresh = r.json()
        assert fresh["run_id"] != original

        # the new dir persists the EDITED document, not the old one
        new_spec = _json.loads(
            (tmp_path / "runs" / fresh["run_id"] / "spec.json").read_text()
        )
        assert new_spec["name"] == "edited-name"
        assert new_spec["motivation"] == "revised motivation"
        # ...while the cancelled history keeps its own spec untouched
        old_spec = _json.loads((tmp_path / "runs" / original / "spec.json").read_text())
        assert old_spec["name"] != "edited-name"

        parked = _await_status(client, fresh["run_id"], *TERMINAL, "awaiting_approval")
        assert any("edited config" in n for n in parked["notes"])


def test_restart_rejects_an_invalid_spec_with_422(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        original = client.post("/api/runs", json=_spec_body(minimal_spec_dict)).json()[
            "run_id"
        ]
        _await_status(client, original, "awaiting_approval")
        client.post(f"/api/runs/{original}/cancel")

        r = client.post(f"/api/runs/{original}/restart", json={"spec": {}})
        assert r.status_code == 422
        # nothing was created: still exactly one run on disk
        assert len(client.get("/api/runs").json()) == 1


# --------------------------------------------------------------- gateway
def test_gateway_serves_crawl_under_its_prefix(minimal_spec_dict, tmp_path):
    with _make_client(tmp_path) as client:
        assert client.get("/api/healthz").json()["status"] == "ok"
        assert client.get("/api/crawl/healthz").json()["status"] == "ok"
        assert client.get("/api/crawl/jobs").json() == []
        assert "llm" in client.get("/api/crawl/usage").json()
