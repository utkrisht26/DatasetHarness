"""Channel protocol — the common interface for the three acquisition channels.

A channel turns one approved :class:`RankedSource` into zero or more provenance-tagged
:class:`CorpusChunk` s, offloading heavy content to the store. Limits and saturation
are enforced by the orchestrator *around* the channel, not inside it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Protocol, runtime_checkable

from adh.contract import JobSpec, SourceChannel
from adh.corpus import CorpusChunk, CorpusStore
from adh.discovery.models import RankedSource
from adh.provenance import sha256_hex


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


@runtime_checkable
class CrawlGovernor(Protocol):
    """The coverage "Brain" a channel consults while acquiring (design-doc §2).

    The agent owns saturation; the channel only reports what it sees
    (:meth:`observe`) and asks whether to keep going (:meth:`should_stop`). This is
    the decoupled STOP signal: novelty detection stays in the orchestrator, never
    in the crawler (design-doc review goal #1).
    """

    def observe(self, aspect_id: str, text: str) -> None:
        """Record one acquired unit of content toward an aspect's coverage."""
        ...

    def should_stop(self, aspect_id: str) -> bool:
        """Return True once ``aspect_id`` is saturated (stop / flush its frontier)."""
        ...


class NullGovernor:
    """Default no-op governor: observe nothing, never stop (offline/tests)."""

    def observe(self, aspect_id: str, text: str) -> None:  # noqa: D401 - no-op
        return None

    def should_stop(self, aspect_id: str) -> bool:
        return False


@dataclass(frozen=True)
class FetchContext:
    """Everything a channel needs to fetch + offload one source."""

    job_id: str
    store: CorpusStore
    spec: JobSpec
    max_items_per_source: int = 50
    clock: Callable[[], datetime] = field(default=_utcnow)
    # The coverage Brain. Channels report acquired content and may flush their
    # frontier when it says the aspect is saturated. Defaults to a no-op so
    # offline channels and existing callers are unaffected.
    governor: CrawlGovernor = field(default_factory=NullGovernor)
    # MEDIA channel only: where content-addressed clip staging lives (the run's
    # ``media_staging`` dir). None for text jobs / channels that don't stage.
    staging_root: Path | None = None
    # aspect_id -> (topic, *keywords) from the plan: the vocabulary a deep crawl
    # judges followed pages against. Empty when the caller has no plan.
    aspect_terms: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


@runtime_checkable
class Channel(Protocol):
    name: SourceChannel

    def fetch(self, source: RankedSource, ctx: FetchContext) -> list[CorpusChunk]:
        """Acquire content for ``source`` and return offloaded corpus chunks."""
        ...


def make_chunk_id(job_id: str, aspect_id: str, content: str) -> str:
    """Deterministic, content-addressed chunk id (identical content -> same id)."""
    return f"{job_id}:{aspect_id}:{sha256_hex(content)[:12]}"
