"""Corpus store + the context-offload pattern (OUTLINE §1.3).

The store is content-addressed (keyed by SHA-256): identical content fetched twice
is stored once. The :class:`CorpusStore` protocol lets a production deployment swap
in S3/MinIO; :class:`LocalCorpusStore` writes to the local filesystem and is the
default for offline runs and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable

from adh.contract import SourceChannel
from adh.provenance import approx_token_count, sha256_hex

from .models import CorpusChunk

PREVIEW_CHARS = 280


@runtime_checkable
class CorpusStore(Protocol):
    def put(self, job_id: str, content: str) -> str:
        """Store content; return a stable ``content_uri``."""
        ...

    def get(self, content_uri: str) -> str:
        """Read content back by its ``content_uri``."""
        ...

    def exists(self, content_uri: str) -> bool: ...

    def size_mb(self, job_id: str) -> float:
        """Total stored bytes for a job, in MB (for the storage circuit-breaker)."""
        ...


@dataclass
class LocalCorpusStore:
    """Filesystem-backed, content-addressed corpus store."""

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def _blob_dir(self, job_id: str) -> Path:
        d = self.root / job_id / "blobs"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def put(self, job_id: str, content: str) -> str:
        digest = sha256_hex(content)
        path = self._blob_dir(job_id) / f"{digest}.txt"
        if not path.exists():
            path.write_text(content, encoding="utf-8")
        return str(path)

    def get(self, content_uri: str) -> str:
        return Path(content_uri).read_text(encoding="utf-8")

    def exists(self, content_uri: str) -> bool:
        return Path(content_uri).exists()

    def size_mb(self, job_id: str) -> float:
        d = self.root / job_id / "blobs"
        if not d.exists():
            return 0.0
        total = sum(f.stat().st_size for f in d.glob("*.txt"))
        return total / (1024 * 1024)


def offload(
    store: CorpusStore,
    *,
    job_id: str,
    aspect_id: str,
    content: str,
    source_channel: SourceChannel,
    source_url: str | None,
    source_license: str,
    chunk_id: str,
    derived_from: tuple[str, ...] = (),
    discovered_via: str = "",
    fetched_at: datetime | None = None,
) -> CorpusChunk:
    """Offload ``content`` to the store and return a context-safe :class:`CorpusChunk`.

    Only the preview + URI are retained in memory; the full text stays in the store.
    """
    content_uri = store.put(job_id, content)
    digest = sha256_hex(content)
    preview = content.strip()[:PREVIEW_CHARS]
    return CorpusChunk(
        chunk_id=chunk_id,
        job_id=job_id,
        aspect_id=aspect_id,
        source_channel=source_channel,
        source_url=source_url,
        source_license=source_license,
        source_sha256=digest,
        content_uri=content_uri,
        text_preview=preview,
        n_tokens=approx_token_count(content),
        fetched_at=fetched_at or datetime.now(timezone.utc),
        derived_from=derived_from,
        discovered_via=discovered_via,
    )
