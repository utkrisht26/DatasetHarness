"""Pure-python MP4 probing — deterministic container metadata without ffmpeg.

Tests hand-build minimal but structurally real MP4 boxes (ftyp/moov/mvhd/tkhd/
mdat) so the parser is exercised against genuine byte layouts, not mocks.
"""

from __future__ import annotations

import struct

import pytest

from adh.media.probe import MediaProbeError, probe_media, probe_mp4


# ------------------------------------------------------------------ mp4 builder
def _box(btype: bytes, *payloads: bytes) -> bytes:
    body = b"".join(payloads)
    return struct.pack(">I", 8 + len(body)) + btype + body


def _mvhd_v0(duration_s: float, timescale: int = 1000) -> bytes:
    body = bytearray(100)  # fixed v0 payload length
    body[12:16] = struct.pack(">I", timescale)
    body[16:20] = struct.pack(">I", int(duration_s * timescale))
    return _box(b"mvhd", bytes(body))


def _mvhd_v1(duration_s: float, timescale: int = 1000) -> bytes:
    body = bytearray(112)  # fixed v1 payload length (u64 times)
    body[0] = 1  # version 1
    body[20:24] = struct.pack(">I", timescale)
    body[24:32] = struct.pack(">Q", int(duration_s * timescale))
    return _box(b"mvhd", bytes(body))


def _tkhd(width: int, height: int) -> bytes:
    body = bytearray(84)  # fixed v0 payload; width/height are trailing 16.16 fixed
    body[76:80] = struct.pack(">I", int(width * 65536))
    body[80:84] = struct.pack(">I", int(height * 65536))
    trak = _box(b"trak", _box(b"tkhd", bytes(body)))
    return trak


def _mp4_bytes(
    *,
    duration_s: float = 6.0,
    width: int = 1920,
    height: int = 1080,
    moov_last: bool = False,
    version1_mvhd: bool = False,
) -> bytes:
    ftyp = _box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41")
    mvhd = _mvhd_v1(duration_s) if version1_mvhd else _mvhd_v0(duration_s)
    moov = _box(b"moov", mvhd, _tkhd(width, height))
    mdat = _box(b"mdat", b"\x00" * 32)
    parts = [ftyp, mdat, moov] if moov_last else [ftyp, moov, mdat]
    return b"".join(parts)


def _clip_file(tmp_path, data: bytes, name: str = "clip.mp4"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


# ----------------------------------------------------------------------- probe
def test_probe_reads_duration_and_dimensions(tmp_path):
    path = _clip_file(tmp_path, _mp4_bytes(duration_s=6.5, width=1920, height=1080))

    result = probe_mp4(path)

    assert result.duration_s == pytest.approx(6.5, abs=0.01)
    assert result.width == 1920
    assert result.height == 1080


def test_probe_handles_moov_at_end(tmp_path):
    path = _clip_file(tmp_path, _mp4_bytes(moov_last=True, duration_s=3.25))

    result = probe_mp4(path)

    assert result.duration_s == pytest.approx(3.25, abs=0.01)


def test_probe_reads_version1_mvhd_64bit_fields(tmp_path):
    path = _clip_file(tmp_path, _mp4_bytes(version1_mvhd=True, duration_s=12.75))

    assert probe_mp4(path).duration_s == pytest.approx(12.75, abs=0.01)


def test_probe_raises_on_garbage_bytes(tmp_path):
    path = _clip_file(tmp_path, b"this is not a movie file at all........")

    with pytest.raises(MediaProbeError):
        probe_mp4(path)


def test_probe_raises_when_moov_is_absent(tmp_path):
    path = _clip_file(tmp_path, _box(b"ftyp", b"isom") + _box(b"mdat", b"\x00" * 16))

    with pytest.raises(MediaProbeError, match="moov"):
        probe_mp4(path)


def test_probe_media_dispatches_by_extension(tmp_path):
    clip = _clip_file(tmp_path, _mp4_bytes(duration_s=2.0), name="c.mov")
    junk = _clip_file(tmp_path, b"avi-not-supported", name="c.avi")

    assert probe_media(clip).duration_s == pytest.approx(2.0, abs=0.01)
    with pytest.raises(MediaProbeError, match="extension"):
        probe_media(junk)


# ------------------------------------------------------------- image containers
def _png_bytes(width: int = 64, height: int = 32) -> bytes:
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)  # 8-bit truecolor
    chunk = b"IHDR" + ihdr
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + chunk
        + struct.pack(">I", 0)
        + b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )


def _jpeg_bytes(width: int = 48, height: int = 24) -> bytes:
    components = b"\x01\x22\x00\x02\x11\x01\x03\x11\x00"  # 3 dummy component triples
    payload = struct.pack(">BHHB", 8, height, width, 3) + components  # SOF0
    return b"\xff\xd8" + b"\xff\xc0" + struct.pack(">H", len(payload) + 2) + payload + b"\xff\xd9"


def _gif_bytes(width: int = 20, height: int = 10, *, frames: int = 1) -> bytes:
    out = bytearray(b"GIF89a")
    out += struct.pack("<HH", width, height)
    out += bytes([0x70, 0, 0])  # packed: no global color table
    if frames > 1:
        out += b"\x21\xff\x0bNETSCAPE2.0\x03\x01\x00\x00\x00"  # loop ext => animated
    for _ in range(frames):
        out += b"\x2c" + struct.pack("<HHHH", 0, 0, width, height) + b"\x00"
        out += b"\x02\x02\x44\x01\x00"  # LZW min-code size + one sub-block + terminator
    out += b"\x3b"
    return bytes(out)


def _webp_bytes(width: int = 30, height: int = 16) -> bytes:
    payload = bytes([0x10, 0, 0, 0])  # VP8X flags (reserved)
    payload += struct.pack("<I", width - 1)[:3] + struct.pack("<I", height - 1)[:3]
    chunk = b"VP8X" + struct.pack("<I", len(payload)) + payload
    return b"RIFF" + struct.pack("<I", 4 + len(chunk)) + b"WEBP" + chunk


def test_probe_png_reads_ihdr_dimensions(tmp_path):
    path = _clip_file(tmp_path, _png_bytes(320, 240), name="img.png")

    result = probe_media(path)

    assert result.media_type == "png"
    assert (result.width, result.height) == (320, 240)
    assert result.duration_s is None


def test_probe_jpeg_walks_to_the_sof_segment(tmp_path):
    path = _clip_file(tmp_path, _jpeg_bytes(800, 600), name="img.jpg")

    result = probe_media(path)

    assert result.media_type == "jpeg"
    assert (result.width, result.height) == (800, 600)
    assert result.duration_s is None


def test_probe_gif_distinguishes_static_from_animated(tmp_path):
    static = _clip_file(tmp_path, _gif_bytes(frames=1), name="a.gif")
    looping = _clip_file(tmp_path, _gif_bytes(frames=3), name="b.gif")

    for path in (static, looping):
        result = probe_media(path)
        assert result.media_type == "gif"
        assert result.duration_s is None  # never guess durations for gifs

    assert probe_media(static).animated is False
    assert probe_media(looping).animated is True


def test_probe_webp_reads_vp8x_canvas(tmp_path):
    path = _clip_file(tmp_path, _webp_bytes(150, 90), name="pic.webp")

    result = probe_media(path)

    assert result.media_type == "webp"
    assert (result.width, result.height) == (150, 90)


def test_sniffed_magic_beats_the_file_extension(tmp_path):
    mislabeled_png = _clip_file(tmp_path, _png_bytes(), name="photo.mp4")
    mislabeled_mp4 = _clip_file(tmp_path, _mp4_bytes(duration_s=1.5), name="clip.png")

    assert probe_media(mislabeled_png).media_type == "png"
    result = probe_media(mislabeled_mp4)
    assert result.media_type == "video"
    assert result.duration_s == pytest.approx(1.5, abs=0.01)


def test_probe_raises_on_structurally_broken_images(tmp_path):
    cases = [
        (_jpeg_bytes()[:10], "truncated.jpeg"),
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 8, "header-only.png"),
        (_gif_bytes()[:12], "cut.gif"),
        (b"\xff\xd8\xff" + b"no-sof-marker-anywhere", "markerless.jpg"),
    ]
    for data, name in cases:
        with pytest.raises(MediaProbeError):
            probe_media(_clip_file(tmp_path, data, name=name))


def test_probe_result_reports_media_type_for_video(tmp_path):
    path = _clip_file(tmp_path, _mp4_bytes(duration_s=4.0))

    assert probe_media(path).media_type == "video"
