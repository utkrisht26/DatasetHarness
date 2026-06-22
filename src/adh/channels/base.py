"""Channel protocol — the common interface for the three acquisition channels.

A channel turns one approved :class:`RankedSource` into zero or more provenance-tagged
:class:`CorpusChunk` s, offloading heavy content to the store. Limits and saturation
are enforced by the orchestrator *around* the channel, not inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Protocol, runtime_checkable

from adh.contract import JobSpec, SourceChannel
from adh.corpus import CorpusChunk, CorpusStore
from adh.discovery.models import RankedSource
from adh.provenance import sha256_hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class FetchContext:
    """Everything a channel needs to fetch + offload one source."""

    job_id: str
    store: CorpusStore
    spec: JobSpec
    max_items_per_source: int = 50
    clock: Callable[[], datetime] = field(default=_utcnow)


@runtime_checkable
class Channel(Protocol):
    name: SourceChannel

    def fetch(self, source: RankedSource, ctx: FetchContext) -> list[CorpusChunk]:
        """Acquire content for ``source`` and return offloaded corpus chunks."""
        ...


def make_chunk_id(job_id: str, aspect_id: str, content: str) -> str:
    """Deterministic, content-addressed chunk id (identical content -> same id)."""
    return f"{job_id}:{aspect_id}:{sha256_hex(content)[:12]}"
