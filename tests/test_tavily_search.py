"""Tests for the Tavily-backed live web-search adapter.

Fully offline: a fake client factory stands in for the ``tavily`` package, and the
module-level client cache in :mod:`adh.search.tavily` is reset between tests.
"""

from __future__ import annotations

import logging

import pytest

from adh.search.provider import SearchHit
from adh.search.tavily_provider import TavilySearchProvider


class _FakeClient:
    def __init__(self, response=None, error=None):
        self._response = response if response is not None else {"results": []}
        self._error = error
        self.calls: list[tuple[str, dict]] = []

    def search(self, query, **kwargs):
        self.calls.append((query, kwargs))
        if self._error is not None:
            raise self._error
        return self._response


_RESPONSE = {
    "results": [
        {
            "title": "Tokio tutorial",
            "url": "https://tokio.rs/tokio/tutorial",
            "content": "async runtime tutorial with spawn and select macros",
            "published_date": "2024-05-01",
        },
        {
            "url": "https://docs.rs/tokio",
            "content": "API reference",
        },
    ]
}


def test_maps_tavily_results_to_search_hits():
    client = _FakeClient(response=_RESPONSE)
    provider = TavilySearchProvider(client_factory=lambda: client)

    hits = provider.search("tokio async tutorial", limit=2)

    assert hits == [
        SearchHit(
            url="https://tokio.rs/tokio/tutorial",
            title="Tokio tutorial",
            snippet="async runtime tutorial with spawn and select macros",
            published="2024-05-01",
        ),
        SearchHit(
            url="https://docs.rs/tokio",
            title="https://docs.rs/tokio",  # title falls back to the URL
            snippet="API reference",
        ),
    ]
    assert client.calls[0][1]["max_results"] == 2


def test_search_errors_degrade_to_empty_with_warning(caplog):
    client = _FakeClient(error=RuntimeError("rate limited"))
    provider = TavilySearchProvider(client_factory=lambda: client)

    with caplog.at_level(logging.WARNING, logger="adh.search.tavily_provider"):
        hits = provider.search("anything")

    assert hits == []  # search failure must not kill discovery
    assert any("rate limited" in r.message for r in caplog.records)


def test_available_requires_key():
    provider = TavilySearchProvider(api_key="", client_factory=lambda: _FakeClient())
    assert not provider.available

    ok = TavilySearchProvider(api_key="tvly-x", client_factory=lambda: _FakeClient())
    assert ok.available


def test_default_key_comes_from_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-env")
    provider = TavilySearchProvider(client_factory=lambda: _FakeClient())
    assert provider.available and provider.api_key == "tvly-env"


def test_live_ports_prefers_tavily_when_key_present(monkeypatch):
    """_live_ports wires Tavily when a key exists; in-memory stub otherwise."""
    from adh.service import app as service_app

    seen: dict[str, object] = {}

    class _StubProvider:
        def __init__(self, api_key=None, client_factory=None):
            seen["key"] = api_key
            self.available = bool(api_key)

        def search(self, query, *, limit=10):
            return []

    monkeypatch.setattr("adh.search.tavily_provider.TavilySearchProvider", _StubProvider)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    search, _ = service_app._live_ports()
    assert type(search).__name__ == "InMemorySearchProvider"  # no key -> stub

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-test")
    search, _ = service_app._live_ports()
    assert isinstance(search, _StubProvider) and seen["key"] == "tvly-test"


# ------------------------------------------------------------------ tavily.py
class _FakeTavilyModule:
    """Stand-in for the real ``tavily`` package (hermetic tests)."""

    class TavilyClient:  # noqa: D401 - simple callable-shaped fake
        def __init__(self, api_key=None):
            self.api_key = api_key


def test_tavily_tools_client_resolution_falls_back_to_env(monkeypatch):
    """Without the external deepagents_code package the tools still find the key
    via TAVILY_API_KEY; with neither, they report a configuration error."""
    import sys
    import types

    import adh.search.tavily as tavily_mod

    fake_pkg = types.ModuleType("tavily")
    fake_pkg.TavilyClient = _FakeTavilyModule.TavilyClient
    monkeypatch.setitem(sys.modules, "tavily", fake_pkg)
    # simulate deepagents_code being absent
    monkeypatch.setitem(sys.modules, "deepagents_code", None)

    monkeypatch.setattr(tavily_mod, "_tavily_client", tavily_mod._UNSET)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    assert tavily_mod._get_tavily_client() is None

    monkeypatch.setattr(tavily_mod, "_tavily_client", tavily_mod._UNSET)
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-env-key")
    client = tavily_mod._get_tavily_client()
    assert client is not None and client.api_key == "tvly-env-key"
    # reset so other tests are untouched
    monkeypatch.setattr(tavily_mod, "_tavily_client", tavily_mod._UNSET)


@pytest.mark.parametrize("tool_name", ["web_search", "fetch_url"])
def test_tavily_tool_names_are_stable(tool_name):
    """deepagents-talon imports web_search/fetch_url by name — keep the contract."""
    import adh.search.tavily as tavily_mod

    assert hasattr(tavily_mod, tool_name)
