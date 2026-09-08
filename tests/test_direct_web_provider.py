"""Direct-web media provider — SERP → polite HTML → media-tag harvest.

Offline: a stub SearchProvider serves SERP pages and a stub PoliteFetcher serves
canned HTML; the extractor's tag/srcset/og handling is pinned with inline
fixtures (no network, no parser dependency beyond the stdlib).
"""

from __future__ import annotations

import logging

from adh.channels.media.direct_web import DirectWebMediaProvider
from adh.search import SearchHit, SearchProvider


PAGE_URL = "https://photos.example/gallery"

GALLERY_HTML = """
<html><head><title>Beach Photo Gallery</title>
<meta property="og:image" content="https://cdn.example/og-cover.jpg">
</head><body>
  <img src="/img/small.jpg" srcset="/img/mid.jpg 640w, https://cdn.example/big.png 1600w" alt="sunset">
  <video poster="/vid/poster.jpg">
    <source src="/vid/clip.mp4" type="video/mp4">
  </video>
  <a href="/files/reel.mov">download reel</a>
  <img src="data:image/png;base64,AAAA">
  <img src="/img/tiny.avif">
</body></html>
"""


class StubSerp(SearchProvider):
    def __init__(self, pages: list[SearchHit]):
        self.pages = pages
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        self.queries.append(query)
        return self.pages[:limit]


class StubFetcher:
    """PoliteFetcher stand-in: canned HTML per page URL + recorded streams."""

    def __init__(self, pages: dict[str, str | None], asset_bodies: dict[str, bytes] | None = None):
        self.pages = pages
        self.asset_bodies = asset_bodies or {}
        self.streamed: list[str] = []

    def fetch_html(self, url: str, *, max_chars: int = 2_000_000) -> str | None:
        return self.pages.get(url)

    def stream(self, url: str, *, max_bytes: int, content_type_prefixes=None):
        self.streamed.append(url)
        body = self.asset_bodies.get(url)
        if body is None:
            raise OSError(f"no canned body for {url!r}")
        yield body


def _provider(
    pages: dict[str, str | None],
    serp_pages: list[SearchHit] | None = None,
    asset_bodies: dict[str, bytes] | None = None,
    **over,
):
    serp = StubSerp(
        serp_pages or [SearchHit(url=PAGE_URL, title="Beach Photo Gallery", snippet="s")]
    )
    fetcher = StubFetcher(pages, asset_bodies=asset_bodies)
    kwargs = {"search_provider": serp, "fetcher": fetcher, **over}
    return DirectWebMediaProvider(**kwargs), serp, fetcher


# ------------------------------------------------------------------ harvest
def test_harvests_media_tags_with_type_and_resolution():
    provider, _, _ = _provider({PAGE_URL: GALLERY_HTML})

    hits = provider.search("beach photos", limit=20)

    by_url = {h.download_url: h for h in hits}
    assert "https://cdn.example/big.png" in by_url  # srcset largest beats /small.jpg src
    assert "https://photos.example/img/mid.jpg" not in by_url  # losing srcset entry dropped
    assert "https://photos.example/img/small.jpg" not in by_url  # src yields to srcset
    assert "https://cdn.example/og-cover.jpg" in by_url  # og:image
    assert "https://photos.example/vid/poster.jpg" in by_url  # video poster
    assert "https://photos.example/vid/clip.mp4" in by_url  # <source>
    assert "https://photos.example/files/reel.mov" in by_url  # media-extension href

    assert by_url["https://cdn.example/big.png"].media_type == "image"
    assert by_url["https://photos.example/vid/clip.mp4"].media_type == "video"
    assert by_url["https://photos.example/files/reel.mov"].media_type == "video"


def test_data_uris_and_unsupported_extensions_are_skipped():
    provider, _, _ = _provider({PAGE_URL: GALLERY_HTML})

    urls = {h.download_url for h in provider.search("beach photos", limit=20)}

    assert not any(u.startswith("data:") for u in urls)
    assert not any(u.endswith(".avif") for u in urls)


def test_hits_carry_unknown_license_and_page_title_attribution():
    provider, _, _ = _provider({PAGE_URL: GALLERY_HTML})

    hits = provider.search("beach photos", limit=20)

    assert all(h.license == "unknown" for h in hits)  # never invented
    sunset = next(h for h in hits if h.tags and h.tags[0] == "sunset")
    assert "Beach Photo Gallery" in sunset.attribution
    assert sunset.page_url == PAGE_URL


def test_media_ids_are_deterministic_per_asset_url():
    provider, _, _ = _provider({PAGE_URL: GALLERY_HTML})

    first = {h.media_id for h in provider.search("beach photos", limit=20)}
    again = {h.media_id for h in provider.search("beach photos", limit=20)}

    assert first == again


def test_non_html_or_blocked_pages_contribute_nothing(caplog):
    provider, _, _ = _provider({PAGE_URL: None, "https://other.example/x": "<html/>"})

    with caplog.at_level(logging.WARNING):
        hits = provider.search("beach", limit=10)

    assert hits == []
    assert any(
        "skip" in r.getMessage().lower() or "unavailable" in r.getMessage().lower()
        for r in caplog.records
    )


def test_media_types_filter_narrows_the_harvest():
    provider, _, _ = _provider({PAGE_URL: GALLERY_HTML})

    images = provider.search("beach photos", limit=20, media_types=["image"])
    videos = provider.search("beach photos", limit=20, media_types=["video"])

    assert images and all(h.media_type == "image" for h in images)
    assert videos and all(h.media_type == "video" for h in videos)


def test_limit_caps_merged_harvest():
    provider, _, _ = _provider({PAGE_URL: GALLERY_HTML})

    hits = provider.search("beach photos", limit=3)

    assert len(hits) == 3


def test_gif_urls_map_to_the_gif_type():
    html = '<html><title>t</title><img src="/pic.gif"></html>'
    provider, _, _ = _provider({PAGE_URL: html})

    [hit] = provider.search("q", limit=5)

    assert hit.media_type == "gif"


def test_download_streams_from_the_fetcher():
    body = b"\x89PNG\r\n\x1a\n(staged-bytes)"
    provider, _, fetcher = _provider(
        {PAGE_URL: GALLERY_HTML},
        asset_bodies={"https://cdn.example/big.png": body},
    )
    [hit] = (
        h
        for h in provider.search("beach photos", limit=20)
        if h.download_url == "https://cdn.example/big.png"
    )

    assert b"".join(provider.download(hit)) == body
    assert fetcher.streamed == ["https://cdn.example/big.png"]
