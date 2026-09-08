"""yt-dlp video provider — player-embedded sites, DRM boundary untouched.

Owner directive: "if I can view it I can download it, unless it is not encoded
such as Netflix etc." yt-dlp's DEFAULT behavior draws exactly that line — it
raises ``DownloadError("... DRM protected")`` for DRM streams unless the
``allow_unplayable_formats`` bypass is set, and refuses format-less results
unless ``ignore_no_formats_error`` is set. **We never set either flag**; the
constant below exists so tests (and reviewers) can prove it. A DRM refusal or
dead extractor is a loud per-hit skip upstream, never a crash and never a
workaround.

Pure helpers (:func:`_targets_for`, :func:`_progressive_mp4`,
:func:`_hit_from_info`) run without yt-dlp installed; the library imports lazily
inside :class:`YtDlpMediaProvider` behind the ``webvideo`` extra.
"""

from __future__ import annotations

import contextlib
import logging
import tempfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from adh.contract.licenses import normalize_license

from .provider import MediaHit

logger = logging.getLogger(__name__)

#: yt-dlp options that would defeat the DRM/encoding boundary. NEVER put these
#: in any options dict this module builds — tests grep every built dict.
YTDLP_BYPASS_FLAGS = ("allow_unplayable_formats", "ignore_no_formats_error")

_CHUNK_BYTES = 64 * 1024

# Progressive mp4 first: trainers want single-file mp4 (video+audio muxed).
# Then any mp4; then best-available. The bypass flags appear nowhere.
_BASE_OPTS: dict[str, Any] = {
    "quiet": True,
    "no_warnings": True,
    "noplaylist": True,
    "format": "best[ext=mp4][acodec!=none][vcodec!=none]/best[ext=mp4]/best",
}


def _targets_for(query: str, *, limit: int = 10, allow_ytsearch: bool = False) -> list[str]:
    """URL queries go straight to the page; plain text needs ytsearch opt-in."""
    q = query.strip()
    if q.startswith(("http://", "https://")):
        return [q]
    if allow_ytsearch:
        return [f"ytsearch{max(1, limit)}:{q}"]
    return []


def _progressive_mp4(formats: list[dict] | None) -> dict | None:
    """Highest-resolution progressive mp4 (audio+video), or None."""
    best: dict | None = None
    for fmt in formats or []:
        if fmt.get("ext") != "mp4":
            continue
        if not fmt.get("acodec") or fmt.get("acodec") == "none":
            continue
        if not fmt.get("vcodec") or fmt.get("vcodec") == "none":
            continue
        height = fmt.get("height") or 0
        if best is None or height > (best.get("height") or 0):
            best = fmt
    return best


def _hit_from_info(info: dict) -> MediaHit:
    """Map one yt-dlp info dict to a :class:`MediaHit` (metadata-tolerant)."""
    fmt = info.get("_requested_format") or {}
    webpage_url = str(info.get("webpage_url") or "").strip()
    download_url = str(fmt.get("url") or info.get("url") or webpage_url)
    raw_license = str(info.get("license") or "").strip()
    canonical = normalize_license(raw_license) if raw_license else "unknown"
    # Site "licenses" (YouTube ToS etc.) are not redistribution grants: only a
    # string the alias table actually recognizes survives; everything else is
    # honestly unknown and goes through the two-layer policy like open web.
    license_ = canonical if canonical.lower() != raw_license.lower() else "unknown"
    tags = tuple(str(t) for t in (info.get("tags") or [])[:8] if t)
    return MediaHit(
        media_id=str(info.get("id") or ""),
        page_url=webpage_url or download_url,
        download_url=download_url,
        title=str(info.get("title") or ""),
        tags=tags,
        duration_s=float(info.get("duration") or 0.0),
        width=int(info.get("width") or 0),
        height=int(info.get("height") or 0),
        license=license_,
        media_type="video",
    )


@dataclass
class YtDlpMediaProvider:
    """Player-embedded video via the yt-dlp Python API (extra ``webvideo``).

    ``ydl_factory`` is the injection seam for offline tests; production builds
    real ``yt_dlp.YoutubeDL`` objects per call (extractors are stateful).
    """

    ydl_factory: Callable[[dict], Any] | None = None
    allow_ytsearch: bool = False
    tmp_root: str | None = None
    supported_media_types: frozenset[str] = frozenset({"video"})

    # ------------------------------------------------------------------ port
    def search(self, query: str, *, limit: int, media_types=None) -> list[MediaHit]:
        targets = _targets_for(query, limit=limit, allow_ytsearch=self.allow_ytsearch)
        if not targets:
            logger.info(
                "yt-dlp: query %r is not a URL and ytsearch is disabled — nothing to do",
                query,
            )
            return []
        factory = self._factory()
        hits: list[MediaHit] = []
        for target in targets:
            try:
                info = factory(self._ydl_opts()).extract_info(target, download=False)
            except Exception as exc:  # noqa: BLE001 - DownloadError incl. DRM refusals
                logger.warning(
                    "yt-dlp: skipping %s (%s) — DRM refusals and dead extractors "
                    "are loud skips, never bypassed",
                    target,
                    exc,
                )
                continue
            entries = info.get("entries") if isinstance(info, dict) else None
            candidates = [e for e in (entries or []) if e] or (
                [info] if isinstance(info, dict) else []
            )
            for entry in candidates:
                chosen = _progressive_mp4(entry.get("formats"))
                hits.append(_hit_from_info({**entry, "_requested_format": chosen or {}}))
                if len(hits) >= limit:
                    return hits
        return hits

    def download(self, hit: MediaHit) -> Iterator[bytes]:
        """Download via yt-dlp into a temp file, then stream its bytes away.

        ``DownloadError`` (DRM refusal, no usable formats, dead extractor)
        propagates to the channel's per-asset guard — the loud-skip contract.
        """
        tmpdir = (
            Path(self.tmp_root) if self.tmp_root else Path(tempfile.mkdtemp(prefix="adh-ytdlp-"))
        )
        tmpdir.mkdir(parents=True, exist_ok=True)
        factory = self._factory()
        ydl = factory(self._ydl_opts(outtmpl_dir=tmpdir))
        info = ydl.extract_info(hit.page_url, download=True)
        filepath = _filepath_from(info, ydl)
        try:
            with filepath.open("rb") as fh:
                while chunk := fh.read(_CHUNK_BYTES):
                    yield chunk
        finally:
            with contextlib.suppress(OSError):
                filepath.unlink(missing_ok=True)

    # --------------------------------------------------------------- plumbing
    def _factory(self) -> Callable[[dict], Any]:
        if self.ydl_factory is not None:
            return self.ydl_factory
        from yt_dlp import YoutubeDL  # lazy: the webvideo extra

        return YoutubeDL

    @staticmethod
    def _ydl_opts(*, outtmpl_dir: Path | None = None) -> dict:
        opts = dict(_BASE_OPTS)
        if outtmpl_dir is not None:
            opts["outtmpl"] = {"default": str(Path(outtmpl_dir) / "%(id)s.%(ext)s")}
        return opts  # YTDLP_BYPASS_FLAGS must never appear here (see module doc)


def _filepath_from(info: dict, ydl: Any) -> Path:
    """Where did yt-dlp put the file? requested_downloads first, then prepare_filename."""
    for entry in info.get("requested_downloads") or []:
        filepath = entry.get("filepath") or entry.get("filename")
        if filepath:
            return Path(filepath)
    with contextlib.suppress(Exception):
        return Path(ydl.prepare_filename(info))
    raise RuntimeError(f"yt-dlp reported no output file for {info.get('id')!r}")
