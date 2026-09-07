"""Media tooling — container probing without external dependencies.

The probe is deliberately stdlib-only (no ffmpeg/ffprobe requirement): it parses
just enough of the ISO base media file format to answer the questions dataset
quality gates ask — duration, pixel dimensions. Richer probing (codec, fps,
frame extraction) is a designed-in TODO behind :func:`probe_media`'s dispatch.
"""

from __future__ import annotations

from .probe import MediaProbeError, MediaProbeResult, probe_media, probe_mp4

__all__ = ["MediaProbeError", "MediaProbeResult", "probe_media", "probe_mp4"]
