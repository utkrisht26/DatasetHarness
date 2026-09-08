"""Deterministic media packager (media modalities) — staged assets → candidates.

For media jobs there is nothing to synthesize: each Engine A MEDIA chunk already
IS the payload (a licensed, staged asset plus its acquisition metadata). The
packager turns one staged asset into one :class:`Candidate`:

* ``input``  = scene descriptor carrying the asset identity (prompt slot),
* ``output`` = caption cleaned from the source metadata (title/tags) — v1;
  vision-model captioning upgrades this seam later without changing the
  pipeline shape,
* ``media`` / ``staged_path`` = the :class:`MediaRef` + local absolute path that
  verification (deterministic media probe) and delivery layouts consume.

No LLM calls are made: integrity is enforced by the record schema's x-verifier
hook and caption quality by the jury, like every other modality.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from adh.contract import JobSpec, MediaRef
from adh.corpus import CorpusStore
from adh.provenance import sha256_prefixed

from .models import Candidate, SeedPoint, make_candidate_id

logger = logging.getLogger(__name__)

# MediaRef.role when the sidecar lacks an explicit media_type: inferred from the
# staged extension; anything unrecognized keeps the historical video default.
_ROLE_BY_SUFFIX = {
    ".mp4": "video",
    ".mov": "video",
    ".m4v": "video",
    ".jpg": "image",
    ".jpeg": "image",
    ".png": "image",
    ".webp": "image",
    ".gif": "gif",
}


class MediaPackager:
    """One staged clip in, one media-backed candidate out (deterministically)."""

    model_id = "packager:adh"

    def generate(
        self, seed: SeedPoint, *, spec: JobSpec, store: CorpusStore | None = None
    ) -> list[Candidate]:
        clip = Path(seed.grounding_uri) if seed.grounding_uri else None
        if clip is None or not clip.is_file():
            logger.warning(
                "media packager: staged clip missing for seed %s (%s) — skipping",
                seed.seed_id,
                clip,
            )
            return []
        try:
            meta = json.loads(clip.with_name(clip.name + ".meta.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("media packager: unreadable sidecar for %s (%s)", clip.name, exc)
            meta = {}

        # The content-addressed filename IS the asset identity; the sidecar is
        # acquisition-time metadata captured by the channel.
        sha = clip.stem if len(clip.stem) == 64 else (seed.source_sha256 or "")
        stat = clip.stat()
        media = MediaRef(
            role=str(meta.get("media_type") or _ROLE_BY_SUFFIX.get(clip.suffix.lower(), "video")),
            path=f"media/{sha}{clip.suffix.lower()}",
            sha256=sha,
            bytes=stat.st_size,
            duration_s=_opt_float(meta.get("duration_s")),
            width=_opt_int(meta.get("width")),
            height=_opt_int(meta.get("height")),
            format=str(meta.get("format") or clip.suffix.lstrip(".").lower() or "bin"),
            attribution=meta.get("attribution"),
        )
        caption = _caption_from(meta)
        if not caption:
            logger.warning("media packager: no caption metadata for %s — skipping", clip.name)
            return []
        input_text = f"{seed.topic}: asset {sha[:8]}" if seed.topic else f"asset {sha[:8]}"

        return [
            Candidate(
                candidate_id=make_candidate_id(seed.job_id, seed.aspect_id, input_text, caption),
                job_id=seed.job_id,
                aspect_id=seed.aspect_id,
                verifiability_tier=spec.verifiability_tier,
                input=input_text,
                output=caption,
                source_channel=seed.source_channel,
                source_url=meta.get("url") or seed.source_url,
                source_license=meta.get("license") or seed.source_license,
                source_sha256=sha or None,
                derived_from=seed.derived_from,
                model_id=self.model_id,
                prompt_hash=sha256_prefixed(caption),
                generation_params=dict(spec.generation_params),
                pipeline_step="media-packaging",
                language=spec.languages[0] if spec.languages else "en",
                task_category=spec.task_categories[0] if spec.task_categories else None,
                media=media,
                staged_path=str(clip.resolve()),
            )
        ]


def _caption_from(meta: dict) -> str:
    """Cleaned source-metadata caption: title first, then tags (v1 seam)."""
    parts = [str(meta.get("title", "")).strip()]
    tags = [str(t).strip() for t in (meta.get("tags") or []) if str(t).strip()]
    return ", ".join([p for p in parts if p] + tags)


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
