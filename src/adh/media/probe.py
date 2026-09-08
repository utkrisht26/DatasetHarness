"""Pure-python media container probes (stdlib only, no ffmpeg).

Magic bytes pick the parser — the file extension is only a tie-breaker for the
MP4 family (some MOVs open with a ``wide``/``skip`` box rather than ``ftyp``).
Each walk reads exactly what the quality gates need:

* MP4/MOV  → ``mvhd`` timescale+duration (v0 u32 / v1 u64), ``tkhd`` dimensions
* JPEG     → marker walk to the first SOFn segment (skips entropy FF00 stuffing)
* PNG      → leading IHDR width/height
* GIF      → logical-screen descriptor + image-descriptor count (animated flag)
* WebP     → VP8X canvas size, falling back to VP8/VP8L payload headers

Images always yield ``duration_s=None``; GIF animatedness is an explicit
``animated`` bool, never a guessed duration. Fragmented MP4s legitimately yield
no duration; callers treat unknown metadata as fail-closed rejection, not guess.
TODO(ffprobe): optional richer probing when an ffprobe binary is available.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

_MP4_SUFFIXES = {".mp4", ".mov", ".m4v"}

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class MediaProbeError(ValueError):
    """The file is not a parseable media container (or lacks required structures)."""


@dataclass(frozen=True)
class MediaProbeResult:
    """Container metadata as probed from disk."""

    duration_s: float | None
    width: int | None
    height: int | None
    fps: float | None = None  # TODO: stts-derived frame rate
    media_type: str = "video"  # "video" | "jpeg" | "png" | "gif" | "webp"
    animated: bool = False  # GIF-only signal; never inferred for other types


def _sniff(head: bytes) -> str | None:
    """Classify a file by its leading magic bytes; None when unrecognized."""
    if head[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if head[:8] == _PNG_SIGNATURE:
        return "png"
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "webp"
    if head[4:8] == b"ftyp":
        return "video"
    return None


# --------------------------------------------------------------------- mp4/mov
def _iter_boxes(
    buf: bytes, start: int = 0, end: int | None = None
) -> Iterator[tuple[bytes, tuple[int, int]]]:
    """Yield ``(type, payload_span)`` for each sibling box in ``buf[start:end]``."""
    end = len(buf) if end is None else end
    pos = start
    while pos + 8 <= end:
        size, btype = struct.unpack_from(">I4s", buf, pos)
        header = 8
        if size == 1:  # 64-bit largesize follows the type field
            if pos + 16 > end:
                return
            size = struct.unpack_from(">Q", buf, pos + 8)[0]
            header = 16
        elif size == 0:  # box extends to the end of its enclosing container
            size = end - pos
        if size < header or pos + size > end:
            raise MediaProbeError(f"corrupt box {btype!r} at offset {pos}")
        yield btype, (pos + header, pos + size)
        pos += size


def _parse_mvhd(buf: bytes, span: tuple[int, int]) -> float:
    body = buf[slice(*span)]
    version = body[0]
    if version == 1:
        timescale, duration = struct.unpack_from(">IQ", body, 20)
    else:
        timescale, duration = struct.unpack_from(">II", body, 12)
    if not timescale:
        raise MediaProbeError("mvhd declares a zero timescale")
    return duration / timescale


def _parse_tkhd(buf: bytes, span: tuple[int, int]) -> tuple[int, int]:
    # Width/height are the trailing two 16.16 fixed-point words in every version.
    body = buf[slice(*span)]
    if len(body) < 8:
        raise MediaProbeError("tkhd too short to carry dimensions")
    width_raw, height_raw = struct.unpack_from(">II", body, len(body) - 8)
    return int(width_raw) >> 16, int(height_raw) >> 16


def probe_mp4(path: str | Path) -> MediaProbeResult:
    """Probe an MP4/MOV file for duration and dimensions.

    Raises :class:`MediaProbeError` on any structural doubt — callers must treat
    unprobeable files as invalid rather than shipping them.
    """
    duration: float | None = None
    width: int | None = None
    height: int | None = None

    with Path(path).open("rb") as fh:
        fh.seek(0, 2)
        fsize = fh.tell()
        pos = 0
        while pos + 8 <= fsize:
            fh.seek(pos)
            header = fh.read(16)
            if len(header) < 8:
                break
            size, btype = struct.unpack_from(">I4s", header, 0)
            hlen = 8
            if size == 1:
                if len(header) < 16:
                    raise MediaProbeError(f"truncated 64-bit box header at offset {pos}")
                size = struct.unpack_from(">Q", header, 8)[0]
                hlen = 16
            elif size == 0:
                size = fsize - pos
            if size < hlen or pos + size > fsize:
                raise MediaProbeError(f"corrupt top-level box {btype!r} at offset {pos}")
            if btype == b"moov":
                fh.seek(pos + hlen)  # header read above may have overshot an 8-byte header
                moov = fh.read(size - hlen)  # metadata-sized; safe to slice
                for child, span in _iter_boxes(moov):
                    if child == b"mvhd":
                        duration = _parse_mvhd(moov, span)
                    elif child == b"trak":
                        for grandchild, gspan in _iter_boxes(moov, *span):
                            if grandchild == b"tkhd" and width is None:
                                width, height = _parse_tkhd(moov, gspan)
            pos += size

    if duration is None:
        raise MediaProbeError(f"{Path(path).name}: no readable moov/mvhd (fragmented or corrupt?)")
    return MediaProbeResult(
        duration_s=duration,
        width=width or None,
        height=height or None,
        media_type="video",
    )


# ------------------------------------------------------------------------ jpeg
_SOF_MARKERS = frozenset(
    {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
)
_PAYLOADLESS_MARKERS = frozenset({0x01, *range(0xD0, 0xDA)})  # TEM + RSTn + SOI/EOI


def _probe_jpeg(p: Path) -> MediaProbeResult:
    """Walk JPEG markers to the first start-of-frame segment for the dimensions."""
    dims: tuple[int, int] | None = None
    with p.open("rb") as fh:
        while True:
            byte = fh.read(1)
            if not byte:
                raise MediaProbeError(f"{p.name}: truncated before any SOF segment")
            if byte != b"\xff":
                continue
            while byte == b"\xff":  # collapse fill bytes before the real marker
                byte = fh.read(1)
                if not byte:
                    raise MediaProbeError(f"{p.name}: truncated inside marker padding")
            marker = byte[0]
            if marker in _PAYLOADLESS_MARKERS:
                continue
            seg_len_raw = fh.read(2)
            if len(seg_len_raw) < 2:
                raise MediaProbeError(f"{p.name}: truncated segment header")
            seg_len = struct.unpack(">H", seg_len_raw)[0]
            if seg_len < 2:
                raise MediaProbeError(f"{p.name}: degenerate segment length {seg_len}")
            if marker in _SOF_MARKERS:
                payload = fh.read(min(seg_len - 2, 5))  # precision + height + width
                if len(payload) < 5:
                    raise MediaProbeError(f"{p.name}: truncated SOF segment")
                height, width = struct.unpack_from(">HH", payload, 1)
                dims = (width, height)
                break
            fh.seek(seg_len - 2, 1)  # skip APPn/COM/DHT… payloads wholesale
    if dims is None or not dims[0] or not dims[1]:
        raise MediaProbeError(f"{p.name}: SOF declares no usable dimensions")
    return MediaProbeResult(duration_s=None, width=dims[0], height=dims[1], media_type="jpeg")


# ------------------------------------------------------------------------- png
def _probe_png(p: Path) -> MediaProbeResult:
    """Read width/height from the mandatory leading IHDR chunk."""
    with p.open("rb") as fh:
        if fh.read(8) != _PNG_SIGNATURE:
            raise MediaProbeError(f"{p.name}: missing PNG signature")
        header = fh.read(8)  # chunk length + type
        if len(header) < 8 or header[4:8] != b"IHDR":
            raise MediaProbeError(f"{p.name}: first chunk is not IHDR")
        body = fh.read(13)
        if len(body) < 13:
            raise MediaProbeError(f"{p.name}: truncated IHDR chunk")
    width, height = struct.unpack_from(">II", body, 0)
    if not width or not height:
        raise MediaProbeError(f"{p.name}: IHDR declares a zero dimension")
    return MediaProbeResult(duration_s=None, width=width, height=height, media_type="png")


# ------------------------------------------------------------------------- gif
def _skip_gif_subblocks(fh) -> bytes | None:
    """Consume one length-prefixed sub-block chain; returns its FIRST block's bytes.

    Returns ``None`` on premature EOF (callers fail closed). Only the first block
    is retained — enough to identify application extensions like NETSCAPE2.0.
    """
    size_raw = fh.read(1)
    if not size_raw:
        return None
    size = size_raw[0]
    if size == 0:  # chain terminator
        return b""
    first = fh.read(size)
    if len(first) < size:
        return None
    if _skip_gif_subblocks(fh) is None:
        return None
    return first


def _probe_gif(p: Path) -> MediaProbeResult:
    """Read the logical screen plus image-descriptor count (animated flag only).

    Durations are NEVER synthesized for gifs — an animated gif's per-frame delays
    are not a training-relevant clip length and guessing would be a lie.
    """
    frames = 0
    animated = False
    with p.open("rb") as fh:
        lsd = fh.read(13)
        if len(lsd) < 13 or lsd[:6] not in (b"GIF87a", b"GIF89a"):
            raise MediaProbeError(f"{p.name}: truncated GIF logical-screen descriptor")
        width, height = struct.unpack_from("<HH", lsd, 6)
        packed = lsd[10]
        if packed & 0x80:  # global color table immediately follows the LSD
            fh.seek(3 * (2 ** ((packed & 0x07) + 1)), 1)
        while True:
            block = fh.read(1)
            if not block:
                raise MediaProbeError(f"{p.name}: truncated GIF block stream")
            code = block[0]
            if code == 0x3B:  # trailer
                break
            if code == 0x2C:  # image descriptor
                desc = fh.read(9)
                if len(desc) < 9:
                    raise MediaProbeError(f"{p.name}: truncated image descriptor")
                frames += 1
                flags = desc[8]
                if flags & 0x80:  # local color table
                    fh.seek(3 * (2 ** ((flags & 0x07) + 1)), 1)
                if not fh.read(1):  # LZW minimum code size
                    raise MediaProbeError(f"{p.name}: truncated image data")
                if _skip_gif_subblocks(fh) is None:
                    raise MediaProbeError(f"{p.name}: truncated image data blocks")
            elif code == 0x21:  # extension introducer
                label_raw = fh.read(1)
                if not label_raw:
                    raise MediaProbeError(f"{p.name}: truncated extension block")
                first = _skip_gif_subblocks(fh)
                if first is None:
                    raise MediaProbeError(f"{p.name}: truncated extension blocks")
                if label_raw[0] == 0xFF and first.startswith(b"NETSCAPE2.0"):
                    animated = True  # looping application extension
            # anything else: unknown block — fail closed below via loop EOF/trailer
    if not width or not height:
        raise MediaProbeError(f"{p.name}: GIF declares a zero dimension")
    return MediaProbeResult(
        duration_s=None,
        width=width,
        height=height,
        media_type="gif",
        animated=animated or frames > 1,
    )


# ------------------------------------------------------------------------ webp
def _probe_webp(p: Path) -> MediaProbeResult:
    """Read canvas dimensions from VP8X, or the lossy/lossless payload headers."""
    with p.open("rb") as fh:
        riff = fh.read(12)
        if len(riff) < 12 or riff[:4] != b"RIFF" or riff[8:12] != b"WEBP":
            raise MediaProbeError(f"{p.name}: not a RIFF/WEBP container")
        while True:
            header = fh.read(8)
            if len(header) < 8:
                raise MediaProbeError(f"{p.name}: WebP carries no VP8X/VP8/VP8L chunk")
            fourcc, size = struct.unpack("<4sI", header)
            body_start = fh.tell()
            if fourcc == b"VP8X":
                payload = fh.read(min(size, 10))
                if len(payload) >= 10:
                    width = int.from_bytes(payload[4:7], "little") + 1
                    height = int.from_bytes(payload[7:10], "little") + 1
                    if width and height:
                        return MediaProbeResult(
                            duration_s=None, width=width, height=height, media_type="webp"
                        )
            elif fourcc == b"VP8 ":  # lossy keyframe: sync code then 14-bit dims
                key = fh.read(10)
                if len(key) >= 10 and key[3:6] == b"\x9d\x01\x2a":
                    width = struct.unpack_from("<H", key, 6)[0] & 0x3FFF
                    height = struct.unpack_from("<H", key, 8)[0] & 0x3FFF
                    if width and height:
                        return MediaProbeResult(
                            duration_s=None, width=width, height=height, media_type="webp"
                        )
            elif fourcc == b"VP8L":  # lossless: signature byte then packed 14-bit dims
                bits = fh.read(5)
                if len(bits) >= 5 and bits[0] == 0x2F:
                    raw = int.from_bytes(bits[1:5], "little")
                    width = (raw & 0x3FFF) + 1
                    height = ((raw >> 14) & 0x3FFF) + 1
                    if width and height:
                        return MediaProbeResult(
                            duration_s=None, width=width, height=height, media_type="webp"
                        )
            fh.seek(body_start + size + (size % 2))  # chunks pad to even offsets


# --------------------------------------------------------------------- dispatch
def probe_media(path: str | Path) -> MediaProbeResult:
    """Probe any supported media container, dispatched by magic bytes.

    The extension participates only as an mp4-family tie-break (MOVs may open
    with ``wide``/``skip`` instead of ``ftyp``). Everything mislabeled still
    probes by its true content; anything unrecognized raises.
    """
    p = Path(path)
    with p.open("rb") as fh:
        head = fh.read(64)
    kind = _sniff(head)
    if kind == "jpeg":
        return _probe_jpeg(p)
    if kind == "png":
        return _probe_png(p)
    if kind == "gif":
        return _probe_gif(p)
    if kind == "webp":
        return _probe_webp(p)
    if kind == "video" or p.suffix.lower() in _MP4_SUFFIXES:
        return probe_mp4(p)
    raise MediaProbeError(
        f"{p.name}: unrecognized media (magic {head[:12].hex()!r}), extension {p.suffix!r}"
    )
