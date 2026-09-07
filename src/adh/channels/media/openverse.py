"""Openverse image provider — keyless CC media over api.openverse.org.

Openverse aggregates CC-licensed and public-domain images (including animated
GIFs) with structured license metadata, so hits carry REAL licenses — the
keyless tier that needs no opt-in flag. Anonymous access is capped at one page
of ≤20 results; a key (``OPENVERSE_API_KEY``) raises rate limits but does not
change this adapter's shape.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

from .provider import MediaHit

logger = logging.getLogger(__name__)

_API_PATH = "/v1/images/"
_MAX_PAGE_SIZE = 20  # API ceiling per page; we never auto-paginate
_UA = "Mozilla/5.0 (compatible; ADH/0.1; open-dataset-acquisition)"

# Openverse's no-attribution family maps straight to canonical names; every
# other short form composes f"cc-{license}[-{version}]" which lands on the
# alias table in adh.contract.licenses. Unrecognized shapes stay "unknown".
_NO_ATTRIBUTION_FAMILY = {
    "cc0": "CC0-1.0",
    "pdm": "public-domain",
    "mark": "public-domain",
}


def composed_license(short_form: str, version: str | None) -> str:
    """Openverse ``license``+``license_version`` → an alias-table spelling."""
    short = (short_form or "").strip().lower()
    if not short:
        return "unknown"
    if short in _NO_ATTRIBUTION_FAMILY:
        return _NO_ATTRIBUTION_FAMILY[short]
    base = f"cc-{short}"
    ver = (version or "").strip().lower()
    return f"{base}-{ver}" if ver else base


class _HttpxAdapter:
    """Shapes httpx.Client into the tiny client protocol this provider uses."""

    def __init__(self, client):  # noqa: ANN001 - httpx optional at import time
        self._client = client

    def get(self, path: str, *, params=None, headers=None):
        return self._client.get(path, params=params, headers=headers)

    def stream(self, url: str, headers=None):
        return self._client.stream("GET", url, headers=headers)


def _default_client_factory(api_key: str | None):
    import httpx

    headers = {"User-Agent": _UA}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return _HttpxAdapter(
        httpx.Client(
            base_url="https://api.openverse.org", headers=headers, timeout=30.0
        )
    )


@dataclass
class OpenverseImageProvider:
    """Creative-Commons image search; works with or without an API key."""

    api_key: str | None = None
    client_factory: Callable[[str | None], Any] | None = None
    supported_media_types: frozenset[str] = frozenset({"image"})
    _client: Any = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def from_env(cls) -> "OpenverseImageProvider":
        """Keyless by default; ``OPENVERSE_API_KEY`` raises the rate limits."""
        return cls(api_key=os.environ.get("OPENVERSE_API_KEY") or None)

    def _client_for_call(self):
        if self._client is None:
            factory = self.client_factory or _default_client_factory
            self._client = factory(self.api_key)
        return self._client

    def search(
        self, query: str, *, limit: int, media_types=None
    ) -> list[MediaHit]:
        page_size = max(1, min(limit, _MAX_PAGE_SIZE))
        if limit > _MAX_PAGE_SIZE:
            logger.warning(
                "Openverse serves at most %d results per query; requested %d — "
                "results are truncated (no silent auto-pagination)",
                _MAX_PAGE_SIZE,
                limit,
            )
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        response = self._client_for_call().get(
            _API_PATH, params={"q": query, "page_size": page_size}, headers=headers or None
        )
        response.raise_for_status()
        payload = response.json()

        wanted = frozenset(media_types) if media_types else None
        hits: list[MediaHit] = []
        for item in payload.get("results", []):
            hit = self._hit_from(item)
            if hit is None:
                logger.warning("skipping malformed Openverse item (missing id or url)")
                continue
            if wanted is not None and hit.media_type not in wanted:
                continue
            hits.append(hit)
            if len(hits) >= limit:
                break
        return hits

    def download(self, hit: MediaHit) -> Iterable[bytes]:
        client = self._client_for_call()
        with client.stream(hit.download_url) as response:
            response.raise_for_status()
            yield from response.iter_bytes(chunk_size=64 * 1024)

    # ------------------------------------------------------------------ map
    @staticmethod
    def _hit_from(item: dict) -> MediaHit | None:
        media_id = str(item.get("id") or "").strip()
        download_url = str(item.get("url") or "").strip()
        if not media_id or not download_url:
            return None
        filetype = str(item.get("filetype") or "").strip().lower()
        attribution = str(item.get("attribution") or "").strip() or _fallback_attribution(item)
        tags = tuple(
            str(tag.get("name")).strip()
            for tag in item.get("tags") or []
            if isinstance(tag, dict) and str(tag.get("name") or "").strip()
        )
        return MediaHit(
            media_id=media_id,
            page_url=str(item.get("foreign_landing_url") or download_url),
            download_url=download_url,
            title=str(item.get("title") or "").strip(),
            tags=tags,
            duration_s=0.0,  # stills never invent durations
            width=int(item.get("width") or 0),
            height=int(item.get("height") or 0),
            license=composed_license(
                str(item.get("license") or ""), item.get("license_version")
            ),
            attribution=attribution,
            media_type="gif" if filetype == "gif" else "image",
        )


def _fallback_attribution(item: dict) -> str:
    parts = [str(item.get("title") or "").strip(), str(item.get("creator") or "").strip()]
    joined = " by ".join(p for p in parts if p)
    return f"{joined} (Openverse)" if joined else "Openverse"
