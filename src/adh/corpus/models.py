"""The Raw Corpus chunk — Engine A's output unit, every chunk carrying provenance.

A chunk is a unit of acquired content (one page's ``fit_markdown``, one dataset
row-group, etc.). The full text lives in the corpus store; the chunk holds only a
``content_uri`` + a short ``text_preview`` so heavy payloads never sit in agent
context (the context-offload pattern, OUTLINE §1.3).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from adh.contract import SourceChannel


@dataclass(frozen=True)
class CorpusChunk:
    chunk_id: str
    job_id: str
    aspect_id: str
    source_channel: SourceChannel
    source_url: str | None
    source_license: str
    source_sha256: str
    content_uri: str  # where the full content is offloaded
    text_preview: str  # short, context-safe preview
    n_tokens: int
    fetched_at: datetime
    derived_from: tuple[str, ...] = ()
    discovered_via: str = ""
    # media chunks (MEDIA channel): probe metadata for the staged binary the
    # content_uri points at; zero-valued for plain text chunks.
    media_format: str = ""
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    n_bytes: int = 0

    def to_manifest_row(self) -> dict:
        """Flat JSON row for ``manifest.jsonl`` (Phase 4.1)."""
        return {
            "kind": "chunk",
            "chunk_id": self.chunk_id,
            "job_id": self.job_id,
            "aspect_id": self.aspect_id,
            "source_channel": self.source_channel.value,
            "source_url": self.source_url,
            "source_license": self.source_license,
            "source_sha256": self.source_sha256,
            "content_uri": self.content_uri,
            "n_tokens": self.n_tokens,
            "fetched_at": self.fetched_at.isoformat(),
            "derived_from": list(self.derived_from),
            "discovered_via": self.discovered_via,
            "media_format": self.media_format,
            "duration_s": self.duration_s,
            "width": self.width,
            "height": self.height,
            "n_bytes": self.n_bytes,
        }

    @classmethod
    def from_manifest_row(cls, row: dict, *, text_preview: str = "") -> "CorpusChunk":
        """Reconstruct a chunk from a manifest row (used on resume)."""
        return cls(
            chunk_id=row["chunk_id"],
            job_id=row["job_id"],
            aspect_id=row["aspect_id"],
            source_channel=SourceChannel(row["source_channel"]),
            source_url=row.get("source_url"),
            source_license=row.get("source_license", "unknown"),
            source_sha256=row.get("source_sha256", ""),
            content_uri=row.get("content_uri", ""),
            text_preview=text_preview,
            n_tokens=int(row.get("n_tokens", 0)),
            fetched_at=datetime.fromisoformat(row["fetched_at"]),
            derived_from=tuple(row.get("derived_from", [])),
            discovered_via=row.get("discovered_via", ""),
            media_format=row.get("media_format") or "",
            duration_s=_opt_float(row.get("duration_s")),
            width=_opt_int(row.get("width")),
            height=_opt_int(row.get("height")),
            n_bytes=int(row.get("n_bytes") or 0),
        )


def _opt_float(value) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _opt_int(value) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class SynthesisHandoff:
    """A synthesis-channel aspect handed straight to Engine B (no corpus, §1.7)."""

    job_id: str
    aspect_id: str
    topic: str
    keywords: tuple[str, ...] = field(default_factory=tuple)

    def to_manifest_row(self) -> dict:
        return {
            "job_id": self.job_id,
            "aspect_id": self.aspect_id,
            "topic": self.topic,
            "keywords": list(self.keywords),
            "source_channel": SourceChannel.SYNTHESIS.value,
        }
