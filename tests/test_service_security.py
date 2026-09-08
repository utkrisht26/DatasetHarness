"""Regression tests for the gateway's deployment security perimeter."""

from __future__ import annotations

from argparse import Namespace

from fastapi.testclient import TestClient

from adh import cli
from adh.service.app import create_app
from adh.service.security import API_KEY_ENV, is_loopback_host


class _CrawlService:
    def counts(self) -> dict:
        return {"queued": 0, "running": 0}

    def close(self) -> None:
        pass


class _RunsService:
    def list(self) -> list:
        return []

    def close(self) -> None:
        pass


def _client(*, api_key: str | None) -> TestClient:
    return TestClient(
        create_app(
            crawl_service=_CrawlService(),
            runs_service=_RunsService(),
            api_key=api_key,
        )
    )


def test_bearer_token_protects_api_but_not_liveness() -> None:
    with _client(api_key="test-secret") as client:
        health = client.get("/api/healthz")
        assert health.status_code == 200
        assert health.json()["auth_enabled"] is True

        missing = client.get("/api/runs")
        assert missing.status_code == 401
        assert missing.headers["www-authenticate"] == "Bearer"
        assert client.get("/api/crawl/jobs").status_code == 401
        assert (
            client.get("/api/runs", headers={"Authorization": "Bearer wrong-secret"}).status_code
            == 401
        )

        authorized = client.get("/api/runs", headers={"Authorization": "Bearer test-secret"})
        assert authorized.status_code == 200
        assert authorized.json() == []


def test_api_key_can_come_from_environment(monkeypatch) -> None:
    monkeypatch.setenv(API_KEY_ENV, "environment-secret")
    with _client(api_key=None) as client:
        assert client.get("/api/runs").status_code == 401
        assert (
            client.get(
                "/api/runs", headers={"Authorization": "Bearer environment-secret"}
            ).status_code
            == 200
        )


def test_auth_can_be_disabled_explicitly_even_when_environment_has_key(
    monkeypatch,
) -> None:
    monkeypatch.setenv(API_KEY_ENV, "environment-secret")
    with _client(api_key="") as client:
        assert client.get("/api/runs").status_code == 200
        assert client.get("/api/healthz").json()["auth_enabled"] is False


def test_http_layer_rejects_unsafe_run_id_before_store_access() -> None:
    with _client(api_key="") as client:
        response = client.get("/api/runs/has%20space")
        assert response.status_code == 422


def test_loopback_host_detection_is_conservative() -> None:
    assert is_loopback_host("127.0.0.1")
    assert is_loopback_host("::1")
    assert is_loopback_host("[::1]")
    assert is_loopback_host("localhost")
    assert not is_loopback_host("0.0.0.0")
    assert not is_loopback_host("::")
    assert not is_loopback_host("service.internal")


def test_cli_refuses_unauthenticated_non_loopback_bind(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.delenv(API_KEY_ENV, raising=False)
    args = Namespace(
        host="0.0.0.0",
        port=8000,
        runs_dir=str(tmp_path),
        discovery_model=None,
        no_llm=True,
        allow_unauthenticated=False,
    )

    assert cli._cmd_serve(args) == 2
    assert "refusing unauthenticated service" in capsys.readouterr().err
