"""CompositeMediaProvider — one MEDIA-channel port fanned out over many sources.

The channel takes exactly ONE provider object; the composite makes N backends
(Pexels, Openverse, direct-web, yt-dlp…) look like one: capability fan-out
against the job's wanted media types, merged results in provider order, URL
dedupe before any bytes move, and downloads delegated to the owning backend.
"""

from __future__ import annotations

import logging

import pytest

from adh.channels.media import (
    CompositeMediaProvider,
    InMemoryMediaProvider,
    MediaHit,
)


def _hit(media_type: str = "video", mid: str | None = None, url: str | None = None, **over):
    mid = mid or f"{media_type}-1"
    fields = dict(
        media_id=mid,
        page_url=f"https://example.com/page/{mid}",
        download_url=url or f"https://cdn.example/{mid}.{media_type[:3]}",
        title=f"a {media_type} asset",
        tags=(media_type,),
        media_type=media_type,
        content=b"payload-" + mid.encode(),
    )
    fields.update(over)
    return MediaHit(**fields)


def _provider(*hits, types=("video",)):
    return InMemoryMediaProvider(hits=list(hits), supported_media_types=frozenset(types))


# ------------------------------------------------------------------ capabilities
def test_supported_types_are_the_union():
    comp = CompositeMediaProvider([_provider(types=("video",)), _provider(types=("image",))])

    assert comp.supported_media_types == frozenset({"video", "image"})
    assert CompositeMediaProvider([]).supported_media_types == frozenset()


def test_fans_out_and_merges_hits_in_provider_order():
    comp = CompositeMediaProvider(
        [_provider(_hit(mid="v1")), _provider(_hit(mid="v2"))]
    )

    hits = comp.search("asset", limit=10)

    assert [h.media_id for h in hits] == ["v1", "v2"]


def test_skips_providers_without_a_wanted_capability():
    spy_calls: list[str] = []

    class Spy(InMemoryMediaProvider):
        def search(self, query, *, limit, media_types=None):
            spy_calls.append(query)
            return super().search(query, limit=limit, media_types=media_types)

    comp = CompositeMediaProvider(
        [
            Spy(hits=[_hit(mid="img", media_type="image")],
                supported_media_types=frozenset({"image"})),
            Spy(hits=[_hit(mid="vid")], supported_media_types=frozenset({"video"})),
        ]
    )

    hits = comp.search("asset", limit=10, media_types=["image"])

    assert [h.media_id for h in hits] == ["img"]
    assert len(spy_calls) == 1  # the video-only backend was never queried


def test_filters_stray_hits_to_the_requested_types():
    class Oblivious(InMemoryMediaProvider):
        # backends that can't narrow may ignore media_types (per the port doc);
        # the composite's own filter is what keeps strays out.
        def search(self, query, *, limit, media_types=None):
            return super().search(query, limit=limit)

    messy = Oblivious(
        hits=[_hit(mid="keep-img", media_type="image"), _hit(mid="drop-vid")],
        supported_media_types=frozenset({"image", "video"}),
    )

    hits = CompositeMediaProvider([messy]).search("asset", limit=10, media_types=["image"])

    assert [h.media_id for h in hits] == ["keep-img"]
    assert all(h.media_type == "image" for h in hits)


def test_no_wanted_types_queries_everything_unchanged():
    comp = CompositeMediaProvider([_provider(_hit(mid="v1"))])

    assert [h.media_id for h in comp.search("asset", limit=5)] == ["v1"]


# ----------------------------------------------------------------------- dedupe
def test_duplicate_download_urls_keep_the_first_provider_hit():
    url = "https://cdn.example/same.mp4"
    comp = CompositeMediaProvider(
        [_provider(_hit(mid="first", url=url)), _provider(_hit(mid="second", url=url))]
    )

    hits = comp.search("asset", limit=10)

    assert [h.media_id for h in hits] == ["first"]


def test_url_dedupe_is_scheme_host_insensitive():
    comp = CompositeMediaProvider(
        [
            _provider(_hit(mid="a", url="https://CDN.example/x.mp4")),
            _provider(_hit(mid="b", url="https://cdn.example/x.mp4#frag")),
        ]
    )

    hits = comp.search("asset", limit=10)

    assert [h.media_id for h in hits] == ["a"]


# -------------------------------------------------------------------- delegation
def test_download_delegates_to_the_owning_provider():
    p1 = _provider(_hit(mid="one"))
    p2 = _provider(_hit(mid="two"))
    comp = CompositeMediaProvider([p1, p2])
    found = {h.media_id: h for h in comp.search("asset", limit=10)}

    assert b"".join(comp.download(found["two"])) == b"payload-two"
    assert b"".join(comp.download(found["one"])) == b"payload-one"


def test_download_of_unsourced_hit_fails_loud():
    comp = CompositeMediaProvider([_provider()])

    with pytest.raises(ValueError, match="owning provider"):
        list(comp.download(_hit(mid="ghost")))


# ------------------------------------------------------------------ resilience
def test_one_broken_backend_is_loud_not_fatal(caplog):
    class Boom(InMemoryMediaProvider):
        def search(self, query, *, limit, media_types=None):
            raise RuntimeError("serp exploded")

    healthy = _provider(_hit(mid="ok"))

    with caplog.at_level(logging.WARNING):
        hits = CompositeMediaProvider([Boom(hits=[], supported_media_types={"video"}), healthy]).search(
            "asset", limit=10
        )

    assert [h.media_id for h in hits] == ["ok"]
    assert any("serp exploded" in r.getMessage() for r in caplog.records)


def test_respects_limit_across_providers():
    comp = CompositeMediaProvider(
        [
            _provider(*[_hit(mid=f"a{i}") for i in range(3)]),
            _provider(*[_hit(mid=f"b{i}") for i in range(3)]),
        ]
    )

    hits = comp.search("asset", limit=4)

    assert len(hits) == 4
