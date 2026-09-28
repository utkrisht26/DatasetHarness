"""Live Pexels video provider (optional extra ``adh[media]``, ``PEXELS_API_KEY``).

Pexels clips carry a site-wide permissive license; the API key is free and
required. Quirks handled here:

* Renditions: each video ships several files — we take the widest ``video/mp4``
  (trainers want mp4; quality wins over codec exoticism).
* No per-video title/tags on the API: the search query stands in as the title
  until the VLM-captioning upgrade replaces source-metadata captions entirely.
* Rate limits surface as HTTP errors from httpx and propagate loudly rather
  than being swallowed mid-run.

The HTTP client is injectable (``client_factory``) so tests run fully offline.
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable
from dataclasses import dataclass, field
from typing import Any

from adh.settings import PEXELS_API_KEY_ENV, get_settings

from .provider import MediaHit

_SEARCH_PATH = "/videos/search"


def _default_client_factory(api_key: str) -> Any:
    import httpx  # lazy: only needed for live use (extra ``adh[media]``)

    return httpx.Client(
        base_url="https://api.pexels.com",
        headers={"Authorization": api_key},
        timeout=30.0,
    )


@dataclass
class PexelsMediaProvider:
    """Search + stream Pexels stock videos through an injectable HTTP client."""

    api_key: str
    client_factory: Callable[[str], Any] = field(default=_default_client_factory)
    supported_media_types: frozenset[str] = frozenset({"video"})
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_env(cls, *, client_factory: Callable[[str], Any] | None = None):
        """Build from ``PEXELS_API_KEY``; loud error when the owner hasn't set it."""
        api_key = get_settings().pexels_api_key_value
        if not api_key:
            raise RuntimeError(
                f"{PEXELS_API_KEY_ENV} is not set; get a free key at https://www.pexels.com/api/ "
                "and export it before running media jobs"
            )
        kwargs = {"api_key": api_key}
        if client_factory is not None:
            kwargs["client_factory"] = client_factory
        return cls(**kwargs)

    # ------------------------------------------------------------------ search
    def search(
        self, query: str, *, limit: int, media_types: Collection[str] | None = None
    ) -> list[MediaHit]:
        client = self._ensure_client()
        response = client.get(_SEARCH_PATH, params={"query": query, "per_page": limit})
        response.raise_for_status()
        hits: list[MediaHit] = []
        for video in response.json().get("videos", []):
            rendition = _best_mp4_rendition(video.get("video_files") or [])
            if rendition is None:
                continue  # mp4-only pipeline: skip exotic-codec-only entries
            user = video.get("user") or {}
            page_url = str(video.get("url") or "")
            tags = tuple(str(t) for t in (video.get("tags") or []))
            hits.append(
                MediaHit(
                    media_id=str(video.get("id", "")),
                    page_url=page_url,
                    download_url=str(rendition.get("link") or ""),
                    title=str(video.get("title") or query),  # no title field -> query
                    tags=tags,
                    duration_s=float(video.get("duration") or 0.0),
                    width=int(video.get("width") or 0),
                    height=int(video.get("height") or 0),
                    license="pexels",
                    attribution=f"Pexels: {user.get('name', 'unknown')} ({user.get('url', '')})",
                    media_type="video",
                )
            )
        return hits

    # ---------------------------------------------------------------- download
    def download(self, hit: MediaHit) -> Iterable[bytes]:
        client = self._ensure_client()
        with client.stream("GET", hit.download_url) as response:
            response.raise_for_status()
            for chunk in response.iter_bytes(chunk_size=1 << 16):
                if chunk:
                    yield chunk

    def _ensure_client(self) -> Any:
        if self._client is None:
            self._client = self.client_factory(self.api_key)
        return self._client


def _best_mp4_rendition(files: list[dict]) -> dict | None:
    """The widest ``video/mp4`` rendition, or None when the video has none."""
    mp4s = [f for f in files if f.get("file_type") == "video/mp4"]
    if not mp4s:
        return None
    return max(mp4s, key=lambda f: int(f.get("width") or 0))
