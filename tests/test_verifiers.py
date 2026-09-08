"""Tier-1 deterministic verifiers — real oracles, exercised offline (no model)."""

from __future__ import annotations

import pytest

from adh.contract import VerifierResult, get_verifier_hook, load_record_schema
from adh.verifiers import VerifierRequest, resolve_verifier, run_verifier
from adh.verifiers.code import run_pytest, run_script
from adh.verifiers.math import check_equal, extract_answer
from adh.verifiers.schema import validate_json

# --------------------------------------------------------------------- code (pytest)
GOOD_SOLUTION = "def add(a, b):\n    return a + b\n"
TESTS = "def test_add():\n    assert add(2, 3) == 5\n    assert add(-1, 1) == 0\n"


def test_run_pytest_passes_correct_solution():
    req = VerifierRequest(input="add two ints", output=GOOD_SOLUTION, tests=TESTS)
    result = run_pytest(req, {"timeout_s": 30})
    assert result.passed is True


def test_run_pytest_rejects_wrong_solution():
    wrong = "def add(a, b):\n    return a - b\n"
    result = run_pytest(VerifierRequest(input="", output=wrong, tests=TESTS), {"timeout_s": 30})
    assert result.passed is False


def test_run_pytest_rejects_syntax_error():
    result = run_pytest(
        VerifierRequest(input="", output="def add(a, b) return a+b", tests=TESTS),
        {"timeout_s": 30},
    )
    assert result.passed is False


def test_run_pytest_without_tests_rejects():
    result = run_pytest(VerifierRequest(input="", output=GOOD_SOLUTION), {})
    assert result.passed is False
    assert "no unit tests" in (result.detail or "")


def test_run_pytest_times_out_on_infinite_loop():
    loop = "def add(a, b):\n    while True:\n        pass\n"
    result = run_pytest(VerifierRequest(input="", output=loop, tests=TESTS), {"timeout_s": 2})
    assert result.passed is False
    assert "tim" in (result.detail or "").lower()


def test_run_script_asserts():
    ok = run_script(
        VerifierRequest(input="", output="x = 1 + 1", tests="assert x == 2"), {"timeout_s": 10}
    )
    assert ok.passed is True
    bad = run_script(
        VerifierRequest(input="", output="x = 1 + 1", tests="assert x == 3"), {"timeout_s": 10}
    )
    assert bad.passed is False


# --------------------------------------------------------------------- sandbox hardening
def _rlimit_as_settable() -> bool:
    """True where the OS allows lowering RLIMIT_AS (Linux); macOS refuses it."""
    try:
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        probe = min(soft, 4 * 1024**3)
        resource.setrlimit(resource.RLIMIT_AS, (probe, hard))
        resource.setrlimit(resource.RLIMIT_AS, (soft, hard))  # restore
        return True
    except (ImportError, ValueError, OSError):
        return False


@pytest.mark.skipif(
    not _rlimit_as_settable(),
    reason="OS does not allow lowering RLIMIT_AS (macOS) — cap cannot be enforced here",
)
def test_memory_bomb_is_capped_by_default():
    """RLIMIT_AS is ON without opt-in: a default config still survives an
    address-space bomb raised at solution import time."""
    bomb = "x = [0] * (512 * 1024 * 1024)  # ~4 GiB of pointers\n"
    result = run_pytest(VerifierRequest(input="", output=bomb, tests=TESTS), {"timeout_s": 30})
    assert result.passed is False
    assert "MemoryError" in (result.detail or "")


def test_file_bomb_is_fsize_capped_by_default():
    """RLIMIT_FSIZE bounds what candidate code may write — no multi-GB temp files."""
    bomb = "open('big.bin', 'wb').write(b'0' * (2 * 1024 * 1024 * 1024))\n"
    result = run_script(VerifierRequest(input="", output="", tests=bomb), {"timeout_s": 30})
    assert result.passed is False


NET_PROBE = (
    "import socket\n"
    "try:\n"
    "    socket.create_connection(('203.0.113.1', 80), timeout=1)\n"
    "except OSError as exc:\n"
    "    print('blocked:', exc)\n"
    "else:\n"
    "    print('NETWORK REACHED')\n"
)


def test_outbound_network_is_disabled_by_default():
    """The sandbox bootstrap kills socket creation before pytest/script code runs."""
    result = run_script(VerifierRequest(input="", output="", tests=NET_PROBE), {"timeout_s": 15})
    assert result.passed is True  # the probe itself exits cleanly
    assert "network disabled by ADH verifier sandbox" in (result.detail or "")


def test_allow_network_config_removes_the_guard():
    result = run_script(
        VerifierRequest(input="", output="", tests=NET_PROBE),
        {"timeout_s": 15, "allow_network": True},
    )
    assert "network disabled by ADH verifier sandbox" not in (result.detail or "")


SPAWNING_LOOP = (
    "import subprocess, sys, time\n"
    "subprocess.Popen([sys.executable, '-c', "
    "'import time; time.sleep(200)  # adh-orphan-probe'])\n"
    "while True:\n"
    "    time.sleep(1)\n"
)


def test_timeout_kills_the_whole_process_group():
    """The candidate spawns a 200s-sleeping grandchild before looping forever.
    Killing only the direct child leaves that grandchild running — the verdict
    returns, but the orphan keeps burning the machine. The whole process group
    must die with the timeout."""
    import subprocess as _subprocess
    import time as _time

    result = run_script(VerifierRequest(input="", output="", tests=SPAWNING_LOOP), {"timeout_s": 2})
    assert result.passed is False and "timed out" in (result.detail or "")

    deadline = _time.monotonic() + 5
    probe = _subprocess.run(["pgrep", "-f", "adh-orphan-probe"], capture_output=True, text=True)
    while probe.returncode == 0 and _time.monotonic() < deadline:
        _time.sleep(0.2)
        probe = _subprocess.run(["pgrep", "-f", "adh-orphan-probe"], capture_output=True, text=True)
    assert probe.returncode != 0, "grandchild survived the timeout kill"


# --------------------------------------------------------------------- math (sympy)
@pytest.mark.parametrize(
    "output,reference",
    [
        ("42", "42"),
        ("The answer is 1/2.", "0.5"),
        ("x**2 + 2*x + 1", "(x+1)**2"),
        (r"\boxed{7}", "7"),
        ("answer = 2*3", "6"),
    ],
)
def test_math_symbolic_equality_passes(output, reference):
    assert check_equal(VerifierRequest(input="", output=output, reference=reference), {}).passed


def test_math_rejects_unequal():
    assert not check_equal(VerifierRequest(input="", output="43", reference="42"), {}).passed


def test_math_without_reference_rejects():
    assert not check_equal(VerifierRequest(input="", output="42"), {}).passed


def test_extract_answer_prefers_boxed():
    assert extract_answer(r"work... \boxed{13} done") == "13"


# --------------------------------------------------------------------- schema
def test_schema_valid_json_matches():
    schema = {"type": "object", "required": ["name"], "properties": {"name": {"type": "string"}}}
    result = validate_json(VerifierRequest(input="", output='{"name": "ada"}'), {"schema": schema})
    assert result.passed is True


def test_schema_rejects_non_json():
    assert not validate_json(VerifierRequest(input="", output="not json"), {}).passed


def test_schema_required_keys_without_full_schema():
    result = validate_json(
        VerifierRequest(input="", output='{"a": 1}'), {"required_keys": ["a", "b"]}
    )
    assert result.passed is False
    assert "missing required keys" in (result.detail or "")


# --------------------------------------------------------------------- dispatch layer
def test_run_verifier_dispatches_from_schema_hook():
    """The locked code_record schema names run_pytest — resolve + run it end to end."""
    schema = load_record_schema("schemas/code_record.schema.json")
    hook = get_verifier_hook(schema)
    assert hook is not None and hook["entry"] == "adh.verifiers.code:run_pytest"
    req = VerifierRequest(input="add", output=GOOD_SOLUTION, tests=TESTS)
    result = run_verifier({**hook, "timeout_s": 30}, req)
    assert result.passed is True


def test_run_verifier_rejects_on_bad_entry_without_raising():
    result = run_verifier(
        {"entry": "adh.verifiers.nope:missing"}, VerifierRequest(input="", output="")
    )
    assert isinstance(result, VerifierResult)
    assert result.passed is False
    assert "could not load" in (result.detail or "")


def test_run_verifier_rejects_missing_entry():
    assert run_verifier({}, VerifierRequest(input="", output="")).passed is False


def test_resolve_verifier_rejects_malformed_entry():
    with pytest.raises(ValueError):
        resolve_verifier("no-colon-here")


# --------------------------------------------------------------------- media probe
def _write_clip(tmp_path, duration_s=6.0, width=1920, height=1080):
    """Hand-built minimal MP4 (structurally real boxes; see test_media_probe.py)."""
    import struct

    def box(btype, *payloads):
        body = b"".join(payloads)
        return struct.pack(">I", 8 + len(body)) + btype + body

    def mvhd_v0(dur_s, timescale=1000):
        b = bytearray(100)
        b[12:16] = struct.pack(">I", timescale)
        b[16:20] = struct.pack(">I", int(dur_s * timescale))
        return box(b"mvhd", bytes(b))

    def tkhd(w, h):
        b = bytearray(84)
        b[76:80] = struct.pack(">I", int(w * 65536))
        b[80:84] = struct.pack(">I", int(h * 65536))
        return box(b"trak", box(b"tkhd", bytes(b)))

    data = (
        box(
            b"ftyp",
            b"isom" + struct.pack(">I", 512) + b"isomiso2mp41",
        )
        + box(b"moov", mvhd_v0(duration_s), tkhd(width, height))
        + box(b"mdat", b"\x00" * 32)
    )
    p = tmp_path / "clip.mp4"
    p.write_bytes(data)
    return p


def _media_request(clip_path, *, sha256=None, media=None, extras_over=None):
    import hashlib

    exists = clip_path.is_file()
    media = (
        media
        if media is not None
        else {
            "role": "video",
            "path": "media/x.mp4",
            "sha256": sha256
            or (hashlib.sha256(clip_path.read_bytes()).hexdigest() if exists else "0" * 64),
            "bytes": clip_path.stat().st_size if exists else 0,
            "format": "mp4",
        }
    )
    extras = {"media": media, "media_abs_path": str(clip_path)}
    extras.update(extras_over or {})
    return VerifierRequest(input="scene", output="caption", extras=extras)


def _write_png(tmp_path, width=640, height=480):
    """Hand-built minimal but structurally-real PNG (IHDR only; see test_media_probe)."""
    import struct

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    chunk = b"IHDR" + ihdr
    data = (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", len(ihdr))
        + chunk
        + struct.pack(">I", 0)
        + b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    p = tmp_path / "img.png"
    p.write_bytes(data)
    return p


def _image_request(img_path, *, role="image", fmt="png"):
    import hashlib

    return _media_request(
        img_path,
        media={
            "role": role,
            "path": f"media/x.{fmt}",
            "sha256": hashlib.sha256(img_path.read_bytes()).hexdigest(),
            "bytes": img_path.stat().st_size,
            "format": fmt,
        },
    )


def test_check_media_passes_well_formed_staged_clip(tmp_path):
    from adh.verifiers.media import check_media

    clip = _write_clip(tmp_path)
    result = check_media(_media_request(clip), {"min_duration_s": 2.0, "min_width": 720})

    assert result.passed is True
    assert "1920x1080" in (result.detail or "")


def test_check_media_rejects_sha256_mismatch(tmp_path):
    from adh.verifiers.media import check_media

    clip = _write_clip(tmp_path)
    req = _media_request(clip, sha256="b" * 64)

    result = check_media(req, {})

    assert result.passed is False
    assert "sha256" in (result.detail or "")


def test_check_media_rejects_duration_out_of_bounds(tmp_path):
    from adh.verifiers.media import check_media

    clip = _write_clip(tmp_path, duration_s=6.0)

    assert check_media(_media_request(clip), {"min_duration_s": 10.0}).passed is False
    assert check_media(_media_request(clip), {"max_duration_s": 5.0}).passed is False


def test_check_media_skips_duration_bounds_for_static_images(tmp_path):
    """Duration bounds are temporal-media concerns; a PNG has no duration to bound."""
    from adh.verifiers.media import check_media

    img = _write_png(tmp_path, width=1024, height=768)

    result = check_media(_image_request(img), {"min_duration_s": 999.0})

    assert result.passed is True
    assert "png ok" in (result.detail or "")  # no duration in the detail either


def test_check_media_enforces_min_width_on_images(tmp_path):
    from adh.verifiers.media import check_media

    img = _write_png(tmp_path, width=100, height=80)

    result = check_media(_image_request(img), {"min_width": 720})

    assert result.passed is False
    assert "width" in (result.detail or "")


def test_check_media_rejects_unreadable_duration_when_bounds_set(tmp_path):
    from adh.verifiers.media import check_media

    # a well-formed container reports no duration only when mvhd is absent;
    # garbage bytes exercise the same fail-closed path via probe failure
    junk = tmp_path / "junk.mp4"
    junk.write_bytes(b"garbage")
    req = _media_request(junk)

    result = check_media(req, {"min_duration_s": 2.0})

    assert result.passed is False
    assert "probe" in (result.detail or "").lower()


def test_check_media_rejects_missing_file(tmp_path):
    from adh.verifiers.media import check_media

    req = _media_request(tmp_path / "gone.mp4")

    result = check_media(req, {})

    assert result.passed is False
    assert "missing" in (result.detail or "")


def test_check_media_never_raises_on_bad_extras():
    from adh.verifiers.media import check_media

    bare = VerifierRequest(input="", output="")
    assert check_media(bare, {}).passed is False
    broken = VerifierRequest(
        input="", output="", extras={"media": "not-a-dict", "media_abs_path": 42}
    )
    assert isinstance(check_media(broken, {}).passed, bool)


def test_run_verifier_dispatches_media_hook_end_to_end(tmp_path):
    import hashlib

    clip = _write_clip(tmp_path, duration_s=4.0)
    data = clip.read_bytes()
    hook = {
        "kind": "media_probe",
        "entry": "adh.verifiers.media:check_media",
        "min_duration_s": 2.0,
        "max_duration_s": 30.0,
    }
    req = _media_request(clip, sha256=hashlib.sha256(data).hexdigest())

    result = run_verifier(hook, req)

    assert result.passed is True
