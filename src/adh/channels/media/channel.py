"""The MEDIA channel — acquire licensed media assets into content-addressed staging.

Unlike the text channels, assets never enter the corpus store: each hit streams
through :class:`MediaStagingStore` (budgeted, atomic, sha256-addressed) and the
:class:`CorpusChunk` points at the staged file with its metadata in a sidecar.
Gates enforced here per spec: asset-type match, license compatibility (with the
two-layer ``allow_unknown_licenses`` opt-in), temporal bounds on temporal types
only, orientation, and the storage budget; saturation stays with the governor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from urllib.parse import urlsplit

from adh.channels.base import FetchContext, make_chunk_id
from adh.contract import DiscoveryConstraints, SourceChannel, resolved_media_types
from adh.contract.licenses import is_compatible, normalize_license
from adh.corpus.models import CorpusChunk
from adh.corpus.store import PREVIEW_CHARS
from adh.discovery.models import RankedSource
from adh.provenance import approx_token_count

from .provider import MediaHit, MediaProvider
from .staging import MediaStagingStore, StorageBudgetExceeded

logger = logging.getLogger(__name__)

# fixed-plan seeds name the source scheme explicitly; auto discovery sets
# locator = search query directly.
_LOCATOR_PREFIXES = ("pexels://", "media://")

# Staged file extension + declared format per hit media type. Video stays mp4:
# every video provider targets mp4 renditions (trainers want mp4).
_EXT_FOR_TYPE: dict[str, str] = {
    "video": "mp4",
    "gif": "gif",
    "jpeg": "jpg",
    "png": "png",
    "webp": "webp",
}

# Rendition URL suffixes trusted for coarse "image" hits whose media_type does
# not name the container; anything else (extensionless CDN links) lands on jpg.
_IMAGE_URL_EXTS = frozenset({"jpg", "jpeg", "png", "webp"})


def _ext_for(hit: MediaHit) -> str:
    """Staged extension + declared format for one hit."""
    if hit.media_type == "video":
        return "mp4"
    if hit.media_type != "image":
        return _EXT_FOR_TYPE.get(hit.media_type, "bin")
    path = urlsplit(hit.download_url.strip()).path.rsplit("/", 1)[-1]
    if "." in path and path.rsplit(".", 1)[1].lower() in _IMAGE_URL_EXTS:
        return path.rsplit(".", 1)[1].lower()
    return "jpg"


def _is_temporal(hit: MediaHit) -> bool:
    """Duration-style bounds are meaningful only for temporal media."""
    return hit.media_type == "video" or (hit.duration_s or 0.0) > 0


@dataclass
class MediaChannel:
    """Turn one ranked MEDIA source into staged assets + provenance chunks."""

    provider: MediaProvider
    name: SourceChannel = SourceChannel.MEDIA

    def fetch(self, source: RankedSource, ctx: FetchContext) -> list[CorpusChunk]:
        if ctx.staging_root is None:
            logger.warning(
                "media channel skipped %s: no staging_root in FetchContext", source.source_id
            )
            return []

        query = _query_of(source.candidate.locator)
        aspect_id = source.candidate.aspect_id
        constraints = ctx.spec.discovery.constraints
        wanted = resolved_media_types(ctx.spec.modality, constraints=constraints)
        staging = MediaStagingStore(
            ctx.staging_root, max_bytes=ctx.spec.limits.max_storage_mb * 1024 * 1024
        )
        allow_unknown = ctx.spec.discovery.allow_unknown_licenses

        chunks: list[CorpusChunk] = []
        seen_shas: set[str] = set()
        for hit in self.provider.search(
            query, limit=ctx.max_items_per_source, media_types=tuple(sorted(wanted))
        ):
            if ctx.governor.should_stop(aspect_id):  # flush the frontier mid-source
                logger.info("aspect %r saturated; flushing media frontier", aspect_id)
                break
            if wanted and hit.media_type not in wanted:
                logger.warning(
                    "skipping %s %s: media_type %r not wanted for modality %r (wanted %s)",
                    hit.media_type, hit.media_id, hit.media_type,
                    ctx.spec.modality, sorted(wanted),
                )
                continue
            if not _passes_filters(hit, constraints):
                continue
            normalized = normalize_license(hit.license)
            known_ok = is_compatible(hit.license, ctx.spec.output_license)
            unknown_ok = normalized == "unknown" and allow_unknown
            if not (known_ok or unknown_ok):
                logger.warning(
                    "skipping %s %s: license %r incompatible with output %r%s",
                    hit.media_type, hit.media_id, hit.license, ctx.spec.output_license,
                    " (and unknown licenses are not allowed by this spec)"
                    if normalized == "unknown" else "",
                )
                continue

            ext = _ext_for(hit)
            meta = {
                "media_id": hit.media_id,
                "title": hit.title,
                "tags": list(hit.tags),
                "license": normalized,
                "url": hit.page_url,
                "attribution": hit.attribution,
                "duration_s": hit.duration_s if _is_temporal(hit) else None,
                "width": hit.width,
                "height": hit.height,
                "format": ext,
                "media_type": hit.media_type,
            }
            # download sits inside the guard too: streaming providers raise
            # during acquisition (hotlink 403s), not only during staging.
            try:
                payload = (
                    hit.content if hit.content is not None else self.provider.download(hit)
                )
                clip = staging.stage(ctx.job_id, payload, ext=ext, meta=meta)
            except StorageBudgetExceeded as exc:
                logger.warning("media storage budget reached for job %s: %s", ctx.job_id, exc)
                break
            except Exception as exc:  # noqa: BLE001 - one dead asset ≠ dead acquisition
                # Hotlink 403s, DRM refusals, expired rendition links: loud skip.
                logger.warning(
                    "skipping media %s: acquisition failed (%s)", hit.media_id, exc
                )
                continue

            if clip.sha256 in seen_shas:  # identical bytes re-surfaced: keep one chunk
                continue
            seen_shas.add(clip.sha256)

            preview = _preview(hit)
            chunks.append(
                CorpusChunk(
                    chunk_id=make_chunk_id(ctx.job_id, aspect_id, clip.sha256),
                    job_id=ctx.job_id,
                    aspect_id=aspect_id,
                    source_channel=self.name,
                    source_url=hit.page_url or None,
                    source_license=hit.license,
                    source_sha256=clip.sha256,
                    content_uri=str(clip.path),
                    text_preview=preview,
                    n_tokens=approx_token_count(preview),
                    fetched_at=ctx.clock(),
                    discovered_via=source.candidate.discovered_via,
                    media_format=ext,
                    duration_s=hit.duration_s if _is_temporal(hit) else None,
                    width=hit.width or None,
                    height=hit.height or None,
                    n_bytes=clip.bytes,
                )
            )
            ctx.governor.observe(aspect_id, preview)
        return chunks


def _query_of(locator: str) -> str:
    for prefix in _LOCATOR_PREFIXES:
        if locator.startswith(prefix):
            return locator[len(prefix):].strip()
    return locator.strip()


def _preview(hit: MediaHit) -> str:
    parts = [part for part in (hit.title, ", ".join(hit.tags)) if part]
    return ", ".join(parts)[:PREVIEW_CHARS]


def _passes_filters(hit: MediaHit, c: DiscoveryConstraints) -> bool:
    if _is_temporal(hit):
        if c.media_min_duration_s is not None and hit.duration_s < c.media_min_duration_s:
            return False
        if c.media_max_duration_s is not None and hit.duration_s > c.media_max_duration_s:
            return False
    if c.media_orientation and hit.width and hit.height:
        if _orientation(hit.width, hit.height) != c.media_orientation:
            return False
    return True


def _orientation(width: int, height: int) -> str:
    if width == height:
        return "square"
    return "landscape" if width > height else "portrait"
