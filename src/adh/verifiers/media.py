"""Deterministic media-integrity verifier (``x-verifier`` hook: ``media_probe``).

Media record schemas declare e.g.::

    { "kind": "media_probe", "entry": "adh.verifiers.media:check_media",
      "min_duration_s": 2.0, "max_duration_s": 30.0, "min_width": 720,
      "allowed_formats": ["jpg", "jpeg", "png", "gif", "webp"] }

``check_media`` re-hashes the staged asset, re-probes the container by magic
bytes, and enforces the hook's bounds — so every shipped record's ``MediaRef``
describes a real, intact file rather than trusting acquisition-time metadata.
Duration bounds apply only to temporal media (video, or anything reporting a
duration); resolution/format checks are universal. Like all verifiers it is
adversarially gameable and therefore **rejects on any doubt**; it never raises
into the pipeline (mirrors :func:`adh.verifiers.base.run_verifier`).
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from adh.contract import VerifierResult
from adh.media.probe import MediaProbeError, probe_media

logger = logging.getLogger(__name__)

_HASH_CHUNK = 1 << 16


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(_HASH_CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _bound_fail(detail: str) -> VerifierResult:
    return VerifierResult(passed=False, detail=detail)


def check_media(request, config: dict) -> VerifierResult:
    """Verify one candidate's staged media against the probe hook's bounds."""
    try:
        media = request.oracle("media")
        staged_path = (request.extras or {}).get("media_abs_path")
        if not isinstance(media, dict) or not isinstance(staged_path, str) or not staged_path:
            return _bound_fail(
                "media probe requires extras['media'] (dict) and extras['media_abs_path']"
            )

        clip = Path(staged_path)
        if not clip.is_file():
            return _bound_fail(f"staged clip missing: {staged_path}")

        expected_sha = media.get("sha256")
        actual_sha = _sha256_file(clip)
        if not expected_sha or actual_sha != expected_sha:
            return _bound_fail(
                f"sha256 mismatch: manifest {expected_sha!r} vs staged {actual_sha!r}"
            )

        try:
            probe = probe_media(clip)
        except MediaProbeError as exc:
            return _bound_fail(f"unprobeable container: {exc}")

        duration = probe.duration_s
        # Duration bounds are temporal-media concerns: they apply to video (and any
        # asset that reports a duration at all); still images legitimately carry
        # none and must not fail merely because a video-shaped bound is configured.
        bounds_apply = probe.media_type == "video" or duration is not None
        min_duration = config.get("min_duration_s")
        max_duration = config.get("max_duration_s")
        if bounds_apply:
            if min_duration is not None and (duration is None or duration < float(min_duration)):
                return _bound_fail(
                    f"duration {duration}s below minimum {min_duration}s (or unreadable)"
                )
            if max_duration is not None and duration is not None and duration > float(max_duration):
                return _bound_fail(f"duration {duration}s above maximum {max_duration}s")

        width, height = probe.width, probe.height
        for axis, value, bound_key in (
            ("width", width, "min_width"),
            ("height", height, "min_height"),
        ):
            minimum = config.get(bound_key)
            if minimum is not None and (value is None or value < int(minimum)):
                return _bound_fail(f"{axis} {value} below minimum {minimum} (or unreadable)")

        allowed_formats = config.get("allowed_formats")
        if allowed_formats and media.get("format") not in allowed_formats:
            return _bound_fail(
                f"format {media.get('format')!r} not in allowed {sorted(allowed_formats)}"
            )

        dur_txt = f"{duration:.2f}s " if duration is not None else ""
        return VerifierResult(
            passed=True,
            detail=(
                f"{probe.media_type} ok: {dur_txt}{width or '?'}x{height or '?'} "
                f"{actual_sha[:12]}… ({media.get('format', '')})"
            ),
        )
    except Exception as exc:  # noqa: BLE001 - a throwing verifier is a rejection
        logger.warning("check_media rejected unexpectedly: %s", exc)
        return _bound_fail(f"media probe error: {exc}")
