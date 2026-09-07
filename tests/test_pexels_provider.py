"""Pexels video provider — live MEDIA-channel source (injected fake client).

The real provider talks to ``https://api.pexels.com/videos/search`` over httpx;
every test here injects a fake client so no network and no key are needed.
"""

from __future__ import annotations

import json

import pytest

from adh.channels.media.pexels import PexelsMediaProvider


def _video_payload(**over) -> dict:
    """One Pexels /videos/search hit with two mp4 renditions + a non-mp4."""
    data = {
        "id": 12345,
        "url": "https://www.pexels.com/video/12345/",
        "duration": 12,
        "width": 1920,
        "height": 1080,
        "user": {"name": "A Videographer", "url": "https://www.pexels.com/@avideographer"},
        "video_files": [
            {
                "file_type": "video/mp4",
                "width": 640,
                "height": 360,
                "link": "https://cdn.pexels.com/sd.mp4",
            },
            {
                "file_type": "video/mp4",
                "width": 1920,
                "height": 1080,
                "link": "https://cdn.pexels.com/hd.mp4",
            },
            {
                "file_type": "video/quicktime",
                "width": 3840,
                "height": 2160,
                "link": "https://cdn.pexels.com/pro.mov",
            },
        ],
    }
    data.update(over)
    return data


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakePexelsClient:
    """Records requests; serves canned JSON; streams canned download bytes."""

    def __init__(self, api_key: str, payload: dict):
        self.api_key = api_key
        self.payload = payload
        self.get_calls: list[tuple[str, dict]] = []
        self.stream_calls: list[str] = []

    def get(self, path: str, params=None) -> _FakeResponse:
        self.get_calls.append((path, dict(params or {})))
        return _FakeResponse(self.payload)

    def stream(self, method: str, url: str):
        self.stream_calls.append(url)
        return _FakeStream(self)


class _FakeStream:
    def __init__(self, client: FakePexelsClient):
        self._client = client

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        return None

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, chunk_size: int = 65536):
        body = json.dumps(self._client.payload).encode()
        for i in range(0, len(body), chunk_size):
            yield body[i : i + chunk_size]


def _provider(payload: dict | None = None, **over) -> tuple[PexelsMediaProvider, FakePexelsClient]:
    client = FakePexelsClient("test-key", payload or {"videos": []})
    kwargs = {"api_key": "test-key", "client_factory": lambda key: client, **over}
    return PexelsMediaProvider(**kwargs), client


# ------------------------------------------------------------------- env wiring
def test_from_env_raises_without_key(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="PEXELS_API_KEY"):
        PexelsMediaProvider.from_env()


def test_from_env_uses_key_and_injected_factory(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "live-key")
    seen_keys: list[str] = []
    provider = PexelsMediaProvider.from_env(
        client_factory=lambda key: seen_keys.append(key) or FakePexelsClient(key, {})
    )

    hits = provider.search("anything", limit=3)

    assert seen_keys == ["live-key"]  # client built lazily on first search
    assert hits == []  # empty payload parses cleanly


# ----------------------------------------------------------------------- search
def test_search_parses_hits_and_picks_widest_mp4_rendition():
    provider, client = _provider({"videos": [_video_payload()]})

    [hit] = provider.search("couple dancing", limit=5)

    assert hit.media_id == "12345"
    assert hit.download_url == "https://cdn.pexels.com/hd.mp4"  # widest mp4, not .mov
    assert hit.page_url == "https://www.pexels.com/video/12345/"
    assert hit.duration_s == 12.0
    assert hit.width == 1920 and hit.height == 1080
    path, params = client.get_calls[0]
    assert path == "/videos/search"
    assert params["query"] == "couple dancing" and int(params["per_page"]) == 5


def test_search_sends_the_api_key_via_client_factory():
    provider, client = _provider()

    provider.search("q", limit=1)

    assert client.api_key == "test-key"


def test_videos_without_mp4_rendition_are_skipped():
    payload = {
        "videos": [
            _video_payload(id=1),
            _video_payload(
                id=2,
                video_files=[
                    {"file_type": "video/quicktime", "width": 1920, "height": 1080,
                     "link": "https://cdn.pexels.com/x.mov"},
                ],
            ),
        ]
    }
    provider, _ = _provider(payload)

    hits = provider.search("dance", limit=10)

    assert [h.media_id for h in hits] == ["1"]


def test_hit_license_and_attribution_carry_pexels_provenance():
    provider, _ = _provider({"videos": [_video_payload()]})

    [hit] = provider.search("dance", limit=1)

    assert hit.license == "pexels"
    assert hit.attribution == "Pexels: A Videographer (https://www.pexels.com/@avideographer)"


def test_title_falls_back_to_query_when_pexels_gives_none():
    """Pexels exposes no per-video title; the query stands in until VLM captions."""
    provider, _ = _provider({"videos": [_video_payload()]})

    [hit] = provider.search("couple dancing on the beach", limit=1)

    assert hit.title == "couple dancing on the beach"


# --------------------------------------------------------------------- download
def test_download_streams_bytes_from_the_rendition_url():
    provider, client = _provider({"videos": [_video_payload()]})
    [hit] = provider.search("dance", limit=1)

    chunks = list(provider.download(hit))

    assert b"".join(chunks)  # non-empty payload echoed back
    assert client.stream_calls == ["https://cdn.pexels.com/hd.mp4"]
