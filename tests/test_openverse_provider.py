"""Openverse image provider — keyless CC-media search over api.openverse.org.

Offline: injected fake clients serve canned JSON exactly like the live API;
license composition lands on the alias table's ``cc-*`` spellings.
"""

from __future__ import annotations

import json
import logging


from adh.channels.media.openverse import OpenverseImageProvider
from adh.contract.licenses import is_compatible


def _item(**over) -> dict:
    data = {
        "id": "abc123",
        "title": "A mountain lake",
        "url": "https://api.openverse.org/v1/images/abc123/thumb/",
        "foreign_landing_url": "https://flickr.com/photos/aperson/123",
        "creator": "Ansel Adams",
        "license": "by-sa",
        "license_version": "4.0",
        "attribution": '"A mountain lake" by Ansel Adams is licensed under CC BY-SA 4.0.',
        "width": 1024,
        "height": 768,
        "filetype": "jpg",
        "tags": [{"name": "mountain"}, {"name": "lake"}],
    }
    data.update(over)
    return data


def _payload(items: list[dict], **meta) -> dict:
    out = {"result_count": len(items), "page_count": 1, "results": items}
    out.update(meta)
    return out


class FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class FakeOpenverseClient:
    """Records GETs; serves canned JSON; streams canned download bytes."""

    def __init__(self, payload: dict):
        self.payload = payload
        self.api_key: str | None = None
        self.get_calls: list[tuple[str, dict, dict]] = []
        self.stream_calls: list[str] = []

    def get(self, path: str, *, params=None, headers=None) -> FakeResponse:
        self.get_calls.append((path, dict(params or {}), dict(headers or {})))
        return FakeResponse(self.payload)

    def stream(self, url: str, headers=None):
        self.stream_calls.append(url)
        return _FakeStream(self)


class _FakeStream:
    def __init__(self, client: FakeOpenverseClient):
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


def _provider(payload: dict | None = None, **over) -> tuple[OpenverseImageProvider, FakeOpenverseClient]:
    client = FakeOpenverseClient(payload or _payload([]))
    kwargs = {"api_key": None, "client_factory": lambda key: client, **over}
    provider = OpenverseImageProvider(**kwargs)
    provider._client = client  # deterministic single client under test
    return provider, client


# ------------------------------------------------------------------- search
def test_search_maps_fields_and_media_type():
    provider, _ = _provider(_payload([_item(), _item(id="gif1", filetype="gif")]))

    hits = provider.search("mountain lake", limit=5)

    assert [h.media_id for h in hits] == ["abc123", "gif1"]
    first = hits[0]
    assert first.media_type == "image"
    assert hits[1].media_type == "gif"
    assert first.page_url == "https://flickr.com/photos/aperson/123"
    assert first.download_url.endswith("/thumb/")
    assert first.width == 1024 and first.height == 768
    assert first.tags == ("mountain", "lake")
    assert "Ansel Adams" in first.attribution
    # images never invent durations
    assert first.duration_s == 0.0


def test_composed_license_is_compatible_with_canonical_cc_output():
    provider, _ = _provider(_payload([_item(license="by-sa", license_version="4.0")]))

    [hit] = provider.search("q", limit=1)

    assert hit.license == "cc-by-sa-4.0"
    assert is_compatible(hit.license, "CC-BY-SA-4.0")  # ShareAlike flows to same output
    assert not is_compatible(hit.license, "CC-BY-4.0")  # …and ONLY ShareAlike output


def test_bare_and_public_domain_forms_compose():
    provider, _ = _provider(
        _payload([
            _item(id="bare", license="by", license_version=""),
            _item(id="pdm", license="pdm", license_version=""),
            _item(id="cc0", license="cc0", license_version="1.0"),
        ])
    )

    hits = {h.media_id: h.license for h in provider.search("q", limit=5)}

    assert hits["bare"] == "cc-by"
    assert hits["pdm"] == "public-domain"
    assert hits["cc0"] == "CC0-1.0"


def test_unknown_license_shape_stays_honestly_unknown():
    provider, _ = _provider(_payload([_item(license="", license_version="")]))

    [hit] = provider.search("q", limit=1)

    assert hit.license == "unknown"


def test_keyless_request_sends_no_auth_header():
    provider, client = _provider()

    provider.search("q", limit=3)

    path, params, headers = client.get_calls[0]
    assert path == "/v1/images/"
    assert params["q"] == "q" and params["page_size"] == 3
    assert all("authorization" not in k.lower() for k in headers)


def test_api_key_is_sent_as_a_bearer_token():
    provider, client = _provider(api_key="ov-key")

    provider.search("q", limit=2)

    _, _, headers = client.get_calls[0]
    assert headers.get("Authorization") == "Bearer ov-key"


def test_page_size_caps_at_20_with_loud_truncation_note(caplog):
    provider, client = _provider(_payload([], result_count=200, page_count=10))

    with caplog.at_level(logging.WARNING):
        hits = provider.search("landscape", limit=50)

    assert client.get_calls[0][1]["page_size"] == 20
    assert hits == []  # empty canned page still parses
    assert any("truncat" in r.getMessage().lower() for r in caplog.records)


def test_malformed_items_are_tolerated(caplog):
    payload = _payload([
        _item(),
        {"id": "broken"},  # no url at all
        _item(id="fine2"),
    ])
    provider, _ = _provider(payload)

    with caplog.at_level(logging.WARNING):
        hits = provider.search("q", limit=5)

    assert [h.media_id for h in hits] == ["abc123", "fine2"]
    assert any("skip" in r.getMessage().lower() for r in caplog.records)


# ----------------------------------------------------------------- download
def test_download_streams_bytes_from_the_item_url():
    provider, client = _provider(_payload([_item()]))
    [hit] = provider.search("q", limit=1)

    chunks = list(provider.download(hit))

    assert b"".join(chunks)
    assert client.stream_calls == ["https://api.openverse.org/v1/images/abc123/thumb/"]
