"""Composite media port — many backends, one :class:`MediaProvider`.

The MEDIA channel takes exactly one provider object; this makes N of them
(Pexels + Openverse + direct-web + yt-dlp…) look like one:

* **Capability fan-out** — a backend is queried only when its
  ``supported_media_types`` intersect the job's wanted types (a yt-dlp SERP
  is never spent on an images-only job).
* **Merge in provider order** — earlier providers rank first.
* **URL dedupe before bytes move** — the same rendition surfaced twice keeps
  the first provider's hit; sha-dedup remains downstream in the channel.
* **Delegated downloads** — search records which backend owns each hit.

One backend failing is loud (logged warning), never fatal while others serve.
"""

from __future__ import annotations

import logging
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from urllib.parse import urlsplit

from .provider import MediaHit, MediaProvider

logger = logging.getLogger(__name__)


def _url_key(url: str) -> str:
    """Canonical-ish identity for dedupe: scheme+host lowercased, fragment dropped."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    if not parts.netloc:
        return url.strip()
    return f"{parts.scheme.lower()}://{parts.netloc.lower()}{parts.path}"


@dataclass
class CompositeMediaProvider:
    """Ordered fan-out wrapper; ``providers[0]`` wins ties and dedupe races."""

    providers: list[MediaProvider] = field(default_factory=list)
    # url-key → backend that produced the hit; repopulated on every search.
    _owner_by_url: dict[str, MediaProvider] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @property
    def supported_media_types(self) -> frozenset[str]:
        out: frozenset[str] = frozenset()
        for provider in self.providers:
            out = out | frozenset(getattr(provider, "supported_media_types", frozenset()))
        return out

    def search(
        self, query: str, *, limit: int, media_types: Collection[str] | None = None
    ) -> list[MediaHit]:
        wanted = frozenset(media_types) if media_types else None
        self._owner_by_url.clear()

        merged: list[MediaHit] = []
        seen_urls: set[str] = set()
        for provider in self.providers:
            capable = frozenset(getattr(provider, "supported_media_types", frozenset()))
            if wanted is not None and not (capable & wanted):
                continue  # capability fan-out: don't spend irrelevant backends
            try:
                hits = provider.search(query, limit=limit, media_types=media_types)
            except Exception as exc:  # noqa: BLE001 - one dead source ≠ dead acquisition
                logger.warning(
                    "media provider %s failed for query %r (%s); continuing",
                    type(provider).__name__, query, exc,
                )
                continue
            for hit in hits:
                if wanted is not None and hit.media_type not in wanted:
                    continue
                key = _url_key(hit.download_url)
                if not key or key in seen_urls:
                    continue
                seen_urls.add(key)
                self._owner_by_url[key] = provider
                merged.append(hit)
                if len(merged) >= limit:
                    return merged
        return merged

    def download(self, hit: MediaHit) -> Iterable[bytes]:
        key = _url_key(hit.download_url)
        owner = self._owner_by_url.get(key)
        if owner is None:
            raise ValueError(
                f"no owning provider for {hit.media_id!r} ({hit.download_url!r}) — "
                "download only hits returned by this composite's search"
            )
        return owner.download(hit)
