"""Media source port — where licensed assets come from (Engine A, MEDIA).

A :class:`MediaProvider` answers "find media matching this query" and streams
the bytes; the MEDIA channel owns everything else (type/license/shape gates,
budgeting, staging, provenance). Media-agnostic by design: hits declare their
``media_type`` ("video", "image", "gif") and providers declare which types they
can serve via ``supported_media_types``. Implementations:

* :class:`InMemoryMediaProvider` — deterministic canned hits (offline / tests).
* :class:`adh.channels.media.composite.CompositeMediaProvider` — fans out over
  many backends as one port.
* ``adh.channels.media.pexels.PexelsMediaProvider`` — the live Pexels API
  (optional extra ``adh[media]``), wired in ``adh.runtime.live_channels``.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class MediaHit:
    """One searchable media result: metadata + a downloadable rendition."""

    media_id: str
    page_url: str  # canonical source page (provenance / attribution target)
    download_url: str  # direct rendition link
    title: str
    tags: tuple[str, ...] = ()
    duration_s: float = 0.0  # temporal media only; stills stay 0.0 (never guessed)
    width: int = 0
    height: int = 0
    license: str = "unknown"  # open-web sources legitimately don't know; never invent
    attribution: str = ""
    media_type: str = "video"  # "video" | "image" | "gif"
    # In-memory convenience so offline providers need no transport at all;
    # live providers leave it None and stream via :meth:`MediaProvider.download`.
    content: bytes | None = None


@runtime_checkable
class MediaProvider(Protocol):
    #: Asset types this backend can actually serve — composite fan-out reads it.
    supported_media_types: frozenset[str]

    def search(
        self, query: str, *, limit: int, media_types: Collection[str] | None = None
    ) -> list[MediaHit]:
        """Return up to ``limit`` hits for ``query``, best first.

        ``media_types`` narrows to the job's wanted asset types when given;
        backends that can't narrow may ignore it (the caller filters too).
        """
        ...

    def download(self, hit: MediaHit) -> Iterable[bytes]:
        """Yield the asset's bytes in chunks (transport lives with the provider)."""
        ...


@dataclass
class InMemoryMediaProvider:
    """Deterministic provider over a seeded hit list (offline / tests).

    ``search`` keeps hits whose title/tags share a word with the query
    (case-insensitive); an empty query returns everything. ``media_types``
    narrows the result when given.
    """

    hits: list[MediaHit] = field(default_factory=list)
    supported_media_types: frozenset[str] = frozenset({"video"})

    def search(
        self, query: str, *, limit: int, media_types: Collection[str] | None = None
    ) -> list[MediaHit]:
        words = {w.lower() for w in query.split() if w}
        wanted = frozenset(media_types) if media_types else None
        out: list[MediaHit] = []
        for hit in self.hits:
            if wanted is not None and hit.media_type not in wanted:
                continue
            if not words or self._matches(hit, words):
                out.append(hit)
                if len(out) >= limit:
                    break
        return out

    @staticmethod
    def _matches(hit: MediaHit, words: set[str]) -> bool:
        haystack = f"{hit.title} {' '.join(hit.tags)}".lower().split()
        return any(word in haystack for word in words)

    def download(self, hit: MediaHit) -> Iterable[bytes]:
        if hit.content is None:
            raise FileNotFoundError(f"no in-memory bytes for media {hit.media_id!r}")
        yield hit.content
