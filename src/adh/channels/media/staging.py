"""Content-addressed media staging (Engine A acquisition → Engine B packaging).

Clips land at ``<root>/<job_id>/<sha256>.<ext>`` so identical assets are stored
once no matter which aspect/source found them, each with a ``.meta.json``
sidecar carrying the acquisition metadata (title/tags/license/attribution) the
caption packager later turns into records. Writes are atomic (temp file +
:func:`os.replace`) and re-staging identical content converges to the same
files, so interrupted runs resume safely.

The storage budget (``spec.limits.max_storage_mb``) is enforced at two points:
a cheap pre-flight check when the source size is knowable up front (local
files), and a hard abort mid-stream for chunked sources (downloads) whose total
size only emerges as bytes arrive.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

_HASH_CHUNK = 1 << 16
_TMP_STEM = ".adh-incoming"


class StorageBudgetExceeded(RuntimeError):
    """The clip cannot be staged within the job's storage budget."""


@dataclass(frozen=True)
class StagedClip:
    """One content-addressed asset on local disk."""

    path: Path
    sha256: str
    bytes: int


class MediaStagingStore:
    """Durable, budgeted, content-addressed clip store under one run directory."""

    def __init__(self, root: Path | str, *, max_bytes: int | None = None) -> None:
        self.root = Path(root)
        self.max_bytes = max_bytes

    # ---------------------------------------------------------------- public
    def stage(
        self,
        job_id: str,
        source: Path | bytes | Iterable[bytes],
        *,
        ext: str | None = None,
        meta: dict | None = None,
    ) -> StagedClip:
        """Stage one clip from a path, raw bytes, or a byte-chunk stream.

        ``ext`` defaults to the source suffix for paths; chunked sources must
        pass it explicitly. Returns the content-addressed :class:`StagedClip`.
        """
        chunks: Iterable[bytes]
        if isinstance(source, Path):
            if ext is None:
                ext = source.suffix.lstrip(".") or "bin"
            self._preflight(source.stat().st_size)
            chunks = _read_chunks(source)
        else:
            if isinstance(source, (bytes, bytearray)):
                source = [bytes(source)]
            ext = (ext or "bin").lstrip(".")
            chunks = source

        job_dir = self.root / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        tmp = job_dir / f"{_TMP_STEM}{ext}"
        free_at_start = self._free_bytes()

        digest = hashlib.sha256()
        written = 0
        try:
            with tmp.open("wb") as out:
                for chunk in chunks:
                    written += len(chunk)
                    if self.max_bytes is not None and written > free_at_start:
                        raise StorageBudgetExceeded(
                            f"media staging budget exceeded: {written} bytes streamed "
                            f"with only {free_at_start} free of {self.max_bytes}"
                        )
                    digest.update(chunk)
                    out.write(chunk)
        except BaseException:
            tmp.unlink(missing_ok=True)  # never leave partial bytes behind
            raise

        sha = digest.hexdigest()
        dest = job_dir / f"{sha}.{ext}"
        os.replace(tmp, dest)  # atomic; re-staging identical content overwrites in place
        # Every clip carries a sidecar; re-staging without metadata never clobbers
        # a richer one written by an earlier acquisition.
        sidecar = job_dir / f"{dest.name}.meta.json"
        if meta is not None or not sidecar.exists():
            sidecar.write_text(json.dumps(meta or {}, indent=2))
        return StagedClip(path=dest, sha256=sha, bytes=written)

    def staged_bytes(self, clip: StagedClip) -> bytes:
        return clip.path.read_bytes()

    def read_meta(self, job_id: str, sha256: str) -> dict:
        """The sidecar metadata for a staged clip, or ``{}`` when absent."""
        sidecars = list((self.root / job_id).glob(f"{sha256}.*.meta.json"))
        if not sidecars:
            return {}
        return json.loads(sidecars[0].read_text())

    # --------------------------------------------------------------- internal
    def _usage(self) -> int:
        return sum(
            p.stat().st_size
            for p in self.root.rglob("*")
            if p.is_file() and not p.name.endswith((".meta.json", _TMP_STEM))
        )

    def _free_bytes(self) -> int:
        if self.max_bytes is None:
            return 1 << 62
        return max(self.max_bytes - self._usage(), 0)

    def _preflight(self, incoming: int) -> None:
        if self.max_bytes is not None and incoming > self._free_bytes():
            raise StorageBudgetExceeded(
                f"media staging budget exceeded before copy: {incoming} incoming, "
                f"{self._free_bytes()} free of {self.max_bytes}"
            )


def _read_chunks(path: Path) -> Iterator[bytes]:
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            yield chunk
