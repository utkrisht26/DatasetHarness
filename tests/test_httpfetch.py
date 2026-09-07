"""Shared polite-fetch plumbing — SSRF guards, robots.txt, spacing, size caps.

Offline: fake HTTP clients/resolvers/clocks exercise every policy branch with
zero network. The real stack (httpx + socket DNS) is wired identically at
runtime; only the seams differ.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from adh.net.httpfetch import (
    OversizedAssetError,
    PoliteFetcher,
    SSRFPolicy,
    UrlBlocked,
    UnsupportedContentType,
)


class FakeResponse:
    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        body: bytes = b"",
        chunks: list[bytes] | None = None,
        peername: tuple[str, int] | None = None,
    ):
        self.status_code = status_code
        self.headers = {k.lower(): v for k, v in (headers or {}).items()}
        self._chunks = chunks if chunks is not None else ([body] if body else [])
        self._iterated = False
        self.extensions: dict = {}
        if peername is not None:
            self.extensions["network_stream"] = SimpleNamespace(
                get_extra_info=lambda key, default=None: (
                    peername if key == "peername" else default
                )
            )

    def iter_bytes(self, chunk_size: int = 65536):
        self._iterated = True
        for chunk in self._chunks:
            for i in range(0, len(chunk), chunk_size):
                yield chunk[i : i + chunk_size]


class FakeClient:
    """Routes URLs to canned responses and records every request."""

    def __init__(self, responses: dict[str, FakeResponse]):
        self.responses = responses
        self.calls: list[tuple[str, dict]] = []

    def get(self, url: str, headers: dict | None = None):
        self.calls.append((url, dict(headers or {})))
        hit = self.responses.get(url)
        if hit is None:
            raise AssertionError(f"unexpected request to {url!r}")
        return hit


PUBLIC_IPS = ["93.184.216.34", "2606:2800:220:1:248:1893:25c8:1946"]


class FakeClock:
    """Monotonic clock advanced by the injected sleeper (deterministic spacing)."""

    def __init__(self, start: float = 1000.0):
        self.now = start
        self.sleeps: list[float] = []

    def clock(self) -> float:
        return self.now

    def sleeper(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


ROBOTS_ALLOW_ALL = "User-agent: *\nAllow: /"


def _policy(resolver_map: dict[str, list[str]] | None = None) -> SSRFPolicy:
    return SSRFPolicy(
        resolver=(lambda host: resolver_map.get(host, PUBLIC_IPS))
        if resolver_map is not None
        else (lambda host: PUBLIC_IPS)
    )


def _fetcher(
    client: FakeClient,
    *,
    robots: dict[str, str | None] | None = None,
    policy: SSRFPolicy | None = None,
    **over,
) -> tuple[PoliteFetcher, FakeClient, FakeClock]:
    time = FakeClock()
    robots_map = {"example.com": ROBOTS_ALLOW_ALL} if robots is None else robots
    kwargs = dict(
        client_factory=lambda: client,
        policy=policy or _policy(),
        robots_loader=lambda netloc: robots_map.get(netloc, None),
        clock=time.clock,
        sleeper=time.sleeper,
        min_interval_s=1.0,
    )
    kwargs.update(over)
    return PoliteFetcher(**kwargs), client, time


HTML_HEADERS = {"content-type": "text/html; charset=utf-8"}


# ------------------------------------------------------------------- spacing
def test_same_netloc_requests_are_spaced():
    client = FakeClient(
        {
            "https://example.com/a": FakeResponse(headers=HTML_HEADERS, body=b"<html>a</html>"),
            "https://example.com/b": FakeResponse(headers=HTML_HEADERS, body=b"<html>b</html>"),
        }
    )
    fetcher, _, time = _fetcher(client)

    assert fetcher.fetch_html("https://example.com/a") == "<html>a</html>"
    assert fetcher.fetch_html("https://example.com/b") == "<html>b</html>"
    assert time.sleeps == [1.0]


def test_distinct_netlocs_need_no_spacing():
    client = FakeClient(
        {
            "https://a.example/x": FakeResponse(headers=HTML_HEADERS, body=b"a"),
            "https://b.example/y": FakeResponse(headers=HTML_HEADERS, body=b"b"),
        }
    )
    fetcher, _, time = _fetcher(client)

    fetcher.fetch_html("https://a.example/x")
    fetcher.fetch_html("https://b.example/y")

    assert time.sleeps == []


def test_crawl_delay_is_honored_but_capped():
    client = FakeClient({"https://slow.example/p": FakeResponse(headers=HTML_HEADERS, body=b"x")})
    robots = {"slow.example": "User-agent: *\nCrawl-delay: 30\nAllow: /"}
    fetcher, _, time = _fetcher(client, robots=robots, max_crawl_delay_s=10.0)

    fetcher.fetch_html("https://slow.example/p")

    assert time.sleeps == []  # first request to the netloc needs no wait
    fetcher.fetch_html("https://slow.example/p")
    assert time.sleeps == [10.0]  # 30s demand capped at the configured ceiling


def test_crawl_delay_below_interval_still_applies():
    client = FakeClient({"https://mid.example/p": FakeResponse(headers=HTML_HEADERS, body=b"x")})
    robots = {"mid.example": "User-agent: *\nCrawl-delay: 3\nAllow: /"}
    fetcher, _, time = _fetcher(client, robots=robots, min_interval_s=1.0)

    fetcher.fetch_html("https://mid.example/p")
    fetcher.fetch_html("https://mid.example/p")

    assert time.sleeps == [3.0]


# -------------------------------------------------------------------- robots
def test_disallowed_path_is_skipped_without_a_request():
    client = FakeClient({})
    robots = {"example.com": "User-agent: *\nDisallow: /private/"}
    fetcher, _, _ = _fetcher(client, robots=robots)

    assert fetcher.fetch_html("https://example.com/private/page") is None
    assert client.calls == []


def test_unreachable_robots_defaults_to_allow_all_with_warning(caplog):
    client = FakeClient(
        {"https://noregistry.example/p": FakeResponse(headers=HTML_HEADERS, body=b"<html/>")}
    )
    fetcher, _, _ = _fetcher(client, robots={"example.com": ROBOTS_ALLOW_ALL})

    with caplog.at_level(logging.WARNING):
        out = fetcher.fetch_html("https://noregistry.example/p")

    assert out == "<html/>"
    assert any("robots" in r.getMessage().lower() for r in caplog.records)


# ---------------------------------------------------------------------- ssrf
def test_private_dns_resolution_is_blocked(caplog):
    client = FakeClient({})
    policy = _policy({"intranet.example": ["127.0.0.1"]})
    fetcher, _, _ = _fetcher(client, policy=policy)

    with caplog.at_level(logging.WARNING):
        with pytest.raises(UrlBlocked, match="blocked"):
            fetcher.fetch_html("https://intranet.example/p")

    assert client.calls == []


def test_metadata_ip_and_loopback_are_blocked():
    policy = SSRFPolicy(resolver=lambda host: ["169.254.169.254"])
    with pytest.raises(UrlBlocked):
        policy.validate("http://evil.example/latest/meta-data")


def test_non_http_scheme_is_blocked():
    with pytest.raises(UrlBlocked, match="scheme"):
        SSRFPolicy(resolver=lambda host: PUBLIC_IPS).validate("ftp://example.com/file")


def test_redirect_hop_is_revalidated():
    client = FakeClient(
        {
            "https://open.example/start": FakeResponse(status_code=302, headers={
                "location": "https://rebind.example/next"
            }),
        }
    )

    class RebindingResolver:
        def __call__(self, host: str) -> list[str]:
            return ["93.184.216.34"] if host == "open.example" else ["10.0.0.7"]

    policy = SSRFPolicy(resolver=RebindingResolver())
    fetcher, client, _ = _fetcher(client, policy=policy)

    with pytest.raises(UrlBlocked):
        fetcher.fetch_html("https://open.example/start")


def test_peer_ip_mismatch_blocks_before_body_is_read():
    # DNS said public, the socket connected to an RFC1918 address: rebinding.
    evil = FakeResponse(headers=HTML_HEADERS, body=b"<html>stolen</html>", peername=("192.168.1.10", 443))
    client = FakeClient({"https://sneaky.example/p": evil})
    fetcher, _, _ = _fetcher(client)

    with pytest.raises(UrlBlocked):
        fetcher.fetch_html("https://sneaky.example/p")

    assert evil._iterated is False


# --------------------------------------------------------------- fetch_html
def test_non_html_content_type_is_skipped(caplog):
    client = FakeClient(
        {"https://example.com/api": FakeResponse(headers={"content-type": "application/json"}, body=b"{}")}
    )
    fetcher, _, _ = _fetcher(client)

    with caplog.at_level(logging.WARNING):
        assert fetcher.fetch_html("https://example.com/api") is None
    assert any("content-type" in r.getMessage().lower() for r in caplog.records)


def test_http_error_status_is_skipped(caplog):
    client = FakeClient({"https://example.com/gone": FakeResponse(status_code=404)})
    fetcher, _, _ = _fetcher(client)

    with caplog.at_level(logging.WARNING):
        assert fetcher.fetch_html("https://example.com/gone") is None


def test_fetch_html_caps_the_body(caplog):
    big = b"<html>" + b"x" * 5000 + b"</html>"
    client = FakeClient({"https://example.com/big": FakeResponse(headers=HTML_HEADERS, body=big)})
    fetcher, _, _ = _fetcher(client, max_chars=1000)

    with caplog.at_level(logging.WARNING):
        out = fetcher.fetch_html("https://example.com/big")

    assert out is not None and len(out) <= 1000
    assert any("truncated" in r.getMessage().lower() for r in caplog.records)


# -------------------------------------------------------------------- stream
def test_stream_rejects_oversize_content_length_upfront():
    client = FakeClient(
        {
            "https://cdn.example/huge.mp4": FakeResponse(
                headers={"content-type": "video/mp4", "content-length": "999999"},
                body=b"",
            )
        }
    )
    fetcher, _, _ = _fetcher(client)

    with pytest.raises(OversizedAssetError):
        fetcher.stream("https://cdn.example/huge.mp4", max_bytes=1000)


def test_stream_aborts_when_body_exceeds_the_cap():
    client = FakeClient(
        {
            "https://cdn.example/liar.mp4": FakeResponse(
                headers={"content-type": "video/mp4"},  # no content-length: lie in the body
                chunks=[b"x" * 800, b"x" * 800],
            )
        }
    )
    fetcher, _, _ = _fetcher(client)

    got = bytearray()
    with pytest.raises(OversizedAssetError):
        for chunk in fetcher.stream("https://cdn.example/liar.mp4", max_bytes=1000):
            got.extend(chunk)

    assert len(got) <= 1800  # stopped well before an unbounded read


def test_stream_rejects_disallowed_content_type():
    client = FakeClient(
        {"https://cdn.example/page.html": FakeResponse(headers=HTML_HEADERS, body=b"<html/>")}
    )
    fetcher, _, _ = _fetcher(client)

    with pytest.raises(UnsupportedContentType):
        list(fetcher.stream("https://cdn.example/page.html", max_bytes=1000))


def test_stream_yields_image_bytes():
    client = FakeClient(
        {
            "https://cdn.example/pic.jpg": FakeResponse(
                headers={"content-type": "image/jpeg"}, chunks=[b"ab", b"cd"]
            )
        }
    )
    fetcher, _, _ = _fetcher(client)

    assert b"".join(fetcher.stream("https://cdn.example/pic.jpg", max_bytes=100)) == b"abcd"


def test_stream_respects_robots_disallow():
    client = FakeClient({})
    robots = {"cdn.example": "User-agent: *\nDisallow: /nope"}
    fetcher, _, _ = _fetcher(client, robots=robots)

    with pytest.raises(UrlBlocked):
        list(fetcher.stream("https://cdn.example/nope/asset.mp4", max_bytes=100))

    assert client.calls == []
