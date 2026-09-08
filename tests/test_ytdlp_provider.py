"""yt-dlp provider — player-embedded video without bypassing any DRM boundary.

The pure mapping/decision helpers run unconditionally (no yt-dlp needed); the
integration paths skip when the ``webvideo`` extra is absent, exactly as the
runtime wiring will.
"""

from __future__ import annotations

import logging

import pytest

from adh.channels.media.ytdlp import (
    YTDLP_BYPASS_FLAGS,  # noqa: F401  # existence pinned below
    _hit_from_info,
    _progressive_mp4,
    _targets_for,
    YtDlpMediaProvider,
)


def test_bypass_flags_are_named_and_never_set():
    # The DRM boundary: these yt-dlp options must never appear in any options
    # dict the provider builds. The constant documents them; tests below grep
    # every built options dict against it.
    assert "allow_unplayable_formats" in YTDLP_BYPASS_FLAGS
    assert "ignore_no_formats_error" in YTDLP_BYPASS_FLAGS


# ------------------------------------------------------- pure decision logic
def test_targets_for_url_queries_go_straight_to_the_page():
    assert _targets_for("https://example.com/watch?v=1", limit=3) == [
        "https://example.com/watch?v=1"
    ]


def test_targets_for_plain_queries_need_ytsearch_opt_in():
    assert _targets_for("dancing birds", limit=4) == []
    assert _targets_for("dancing birds", limit=4, allow_ytsearch=True) == [
        "ytsearch4:dancing birds"
    ]


def test_progressive_mp4_prefers_highest_resolution():
    formats = [
        {
            "format_id": "low",
            "ext": "mp4",
            "acodec": "mp4a.40.2",
            "vcodec": "avc1.64001f",
            "height": 360,
        },
        {
            "format_id": "high",
            "ext": "mp4",
            "acodec": "mp4a.40.2",
            "vcodec": "avc1.640028",
            "height": 1080,
        },
        {
            "format_id": "video-only",
            "ext": "webm",
            "acodec": "none",
            "vcodec": "vp9",
            "height": 2160,
        },
    ]

    assert _progressive_mp4(formats)["format_id"] == "high"


def test_progressive_mp4_returns_none_without_a_usable_format():
    assert _progressive_mp4([{"format_id": "drm-only", "has_drm": True}]) is None
    assert _progressive_mp4([]) is None


def test_hit_from_info_maps_the_info_dict():
    info = {
        "id": "abc123",
        "webpage_url": "https://video.example/watch?v=abc123",
        "title": "A dancing bird",
        "duration": 15.5,
        "width": 1920,
        "height": 1080,
        "tags": ["birds", "dance"],
        "license": "Some Site Terms",
        "_requested_format": {"url": "https://cdn.video.example/abc.mp4", "ext": "mp4"},
    }

    hit = _hit_from_info(info)

    assert hit.media_id == "abc123"
    assert hit.page_url == "https://video.example/watch?v=abc123"
    assert hit.download_url == "https://cdn.video.example/abc.mp4"
    assert hit.title == "A dancing bird"
    assert hit.duration_s == 15.5
    assert hit.media_type == "video"
    assert hit.license == "unknown"  # site ToS is not a redistribution license


def test_hit_from_info_defaults_missing_metadata():
    hit = _hit_from_info({"id": "x", "webpage_url": "https://v.example/x"})

    assert hit.title == "" and hit.duration_s == 0.0
    assert hit.license == "unknown"
    assert hit.download_url == "https://v.example/x"


# ------------------------------------------------------------ integration
yt_dlp = pytest.importorskip("yt_dlp", reason="the webvideo extra is not installed")

INFO = {
    "id": "vid1",
    "webpage_url": "https://video.example/watch?v=vid1",
    "title": "Clip one",
    "duration": 8.0,
    "width": 1280,
    "height": 720,
    "formats": [
        {
            "format_id": "18",
            "ext": "mp4",
            "acodec": "mp4a.40.2",
            "vcodec": "avc1.42001E",
            "height": 360,
            "url": "https://cdn.example/f18.mp4",
        },
        {
            "format_id": "22",
            "ext": "mp4",
            "acodec": "mp4a.40.2",
            "vcodec": "avc1.64001F",
            "height": 720,
            "url": "https://cdn.example/f22.mp4",
        },
    ],
}


class FakeYoutubeDL:
    """Scripted YoutubeDL stand-in capturing every options dict it is built with."""

    built_options: list[dict] = []

    def __init__(self, options: dict):
        self.options = options
        FakeYoutubeDL.built_options.append(dict(options))
        self.downloaded_files: list[str] = []

    def extract_info(self, url: str, download: bool = False):
        if isinstance(INFO.get(url), Exception):
            raise INFO[url]
        return INFO


def _provider(**over) -> YtDlpMediaProvider:
    kwargs = {"ydl_factory": FakeYoutubeDL, **over}
    return YtDlpMediaProvider(**kwargs)


def test_search_never_sets_the_bypass_flags():
    provider = _provider()

    provider.search("https://video.example/watch?v=vid1", limit=1)

    for options in FakeYoutubeDL.built_options:
        for flag in YTDLP_BYPASS_FLAGS:
            assert not options.get(flag), f"DRM boundary violated via {flag}"


def test_search_maps_a_direct_url_to_a_hit():
    provider = _provider()

    [hit] = provider.search("https://video.example/watch?v=vid1", limit=1)

    assert hit.media_id == "vid1"
    assert hit.download_url == "https://cdn.example/f22.mp4"  # best progressive mp4
    assert hit.duration_s == 8.0


def test_drm_error_is_loud_skip_not_crash(caplog):
    drm_info = {
        "id": "locked",
        "webpage_url": "https://video.example/watch?v=locked",
        "title": "Locked",
        "formats": [],
    }
    original_extract = FakeYoutubeDL.extract_info

    def locked(self, url: str, download: bool = False):
        if "locked" in url:
            raise yt_dlp.utils.DownloadError("This video is DRM protected")
        return drm_info

    FakeYoutubeDL.extract_info = locked
    try:
        provider = _provider()
        with caplog.at_level(logging.WARNING):
            hits = provider.search("https://video.example/watch?v=locked", limit=3)
    finally:
        FakeYoutubeDL.extract_info = original_extract

    assert hits == []
    assert any(
        "skip" in r.getMessage().lower() or "drm" in r.getMessage().lower() for r in caplog.records
    )


def test_download_streams_bytes_and_cleans_up(tmp_path):
    staged_file = tmp_path / "vid1.mp4"
    staged_file.write_bytes(b"\x00\x00\x00\x18ftypisom(fake-ytdlp-bytes)")

    def download_writes_file(self, url: str, download: bool = False):
        return dict(INFO, requested_downloads=[{"filepath": str(staged_file)}])

    original_extract = FakeYoutubeDL.extract_info
    FakeYoutubeDL.extract_info = download_writes_file
    try:
        provider = _provider()
        [hit] = provider.search("https://video.example/watch?v=vid1", limit=1)
        chunks = list(provider.download(hit))
    finally:
        FakeYoutubeDL.extract_info = original_extract

    assert b"".join(chunks).startswith(b"\x00\x00\x00\x18ftyp")
    assert not staged_file.exists()  # temp file removed after streaming
