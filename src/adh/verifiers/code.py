"""Tier-1 code verifier — run candidate code against unit tests in a sandbox.

This is the oracle the locked contract already names
(``adh.verifiers.code:run_pytest`` in ``schemas/code_record.schema.json``). It runs
the candidate ``output`` together with the candidate's ``tests`` in an isolated
subprocess with:

* a wall-clock timeout and a CPU-time ``RLIMIT`` backstop;
* an address-space cap (``RLIMIT_AS``, default 512 MiB, ``mem_mb: 0`` opts out)
  — enforced where the OS allows lowering it (Linux); macOS refuses
  ``setrlimit(RLIMIT_AS)``, so the cap is silently skipped there;
* a file-size cap (``RLIMIT_FSIZE``, default 64 MiB, ``max_file_mb: 0`` opts out)
  so candidate code cannot flood the disk;
* outbound networking disabled by default (``allow_network: True`` restores it):
  a ``sitecustomize.py`` bootstrap on ``PYTHONPATH`` kills socket creation *and*
  name resolution before candidate code runs — DNS is an exfiltration channel;
* its own process group (``Popen(process_group=0)``), so the timeout kill takes
  spawned children too — no orphaned grandchildren survive the verdict.

Sandbox honesty: this is a *floor*, not a jail. It bounds runaway time, memory,
disk, and casual network use of Python-level candidate code; native extensions or
ctypes can still reach through. Production should run this inside a container /
nsjail / seccomp profile; the entry point and calling convention stay identical,
so that hardening is a deployment concern, not a code change.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

from adh.contract import VerifierResult

from .base import VerifierRequest, tail

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 10.0
_DEFAULT_MEM_MB = 512
_DEFAULT_MAX_FILE_MB = 64

# Loaded by the child interpreter during startup because its directory is the
# only PYTHONPATH entry. Python-level guard: blocks every stdlib/networking
# library's path in (they all construct ``socket.socket`` eventually) plus the
# DNS-resolving calls. ``socket.socket`` is replaced by a *class* raising in
# ``__init__`` — the stdlib ``ssl`` module subclasses ``socket.socket``, so a
# plain-function replacement would crash unrelated imports at class-definition
# time. ``socketpair`` stays available: loopback IPC is not outbound networking,
# and async frameworks legitimately need it.
_SITECUSTOMIZE = """\
# ADH verifier sandbox bootstrap: outbound networking is disabled for candidate
# code — sockets cannot be created, and name resolution fails (DNS is an
# exfiltration channel). Set x-verifier config "allow_network": true to restore.
import socket


def _adh_network_blocked(*_args, **_kwargs):
    raise OSError("network disabled by ADH verifier sandbox")


class _adh_blocked_socket(socket.socket):
    def __init__(self, *args, **kwargs):
        raise OSError("network disabled by ADH verifier sandbox")


socket.socket = _adh_blocked_socket
socket.SocketType = _adh_blocked_socket
socket.create_connection = _adh_network_blocked
socket.getaddrinfo = _adh_network_blocked  # type: ignore[assignment]
socket.gethostbyname = _adh_network_blocked
socket.gethostbyname_ex = _adh_network_blocked
socket.gethostbyaddr = _adh_network_blocked
"""


def _rlimits(cpu_seconds: int, mem_mb: int, max_file_mb: int):
    """POSIX ``preexec_fn`` capping CPU time, address space, and written-file size."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None

    def _apply() -> None:
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1))
        except (ValueError, OSError):  # pragma: no cover - platform dependent
            pass
        if mem_mb > 0:
            n = mem_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_AS, (n, n))
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass
        if max_file_mb > 0:
            n = max_file_mb * 1024 * 1024
            try:
                resource.setrlimit(resource.RLIMIT_FSIZE, (n, n))
            except (ValueError, OSError):  # pragma: no cover - platform dependent
                pass

    return _apply


def _sandbox_env(tmp_dir: Path, *, allow_network: bool) -> dict[str, str]:
    """Scrubbed child environment; points PYTHONPATH at the sandbox bootstrap.

    The pytest oracle also disables plugin autoloading: installed plugins (an
    IDE/langchain toolchain can register ``pytest11`` entry points) would drag
    unrelated heavyweight imports into every sandboxed child — attack surface
    and startup cost the oracle does not need. Candidate tests are plain
    asserts against ``solution.py``.
    """
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"}
    if not allow_network:
        env["PYTHONPATH"] = str(tmp_dir)  # sitecustomize.py lives here
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    return env


def _kill_group(proc: subprocess.Popen) -> None:
    """SIGKILL the child's whole process group (the child leads it via process_group=0).

    Killing only the direct pid would orphan any grandchild the candidate spawned
    (test subprocesses, sleepers) — they hold inherited fds and keep running.
    """
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):  # pragma: no cover - already gone
        proc.kill()


def run_pytest(request: VerifierRequest, config: dict) -> VerifierResult:
    """Execute ``request.tests`` against ``request.output`` with pytest; pass iff green.

    The solution is written to ``solution.py`` and the tests to a test module that
    first ``from solution import *`` — so tests reference the solution's names
    directly. Config keys: ``timeout_s`` (wall-clock), ``mem_mb`` (address-space
    cap, MiB, 0 = uncapped), ``max_file_mb`` (written-file cap, MiB, 0 = uncapped),
    ``allow_network`` (default False).
    """
    tests = request.oracle("tests")
    if not tests:
        return VerifierResult(passed=False, detail="no unit tests supplied for code verification")

    timeout_s = float(config.get("timeout_s", _DEFAULT_TIMEOUT_S))
    mem_mb = int(config.get("mem_mb", _DEFAULT_MEM_MB))
    max_file_mb = int(config.get("max_file_mb", _DEFAULT_MAX_FILE_MB))
    allow_network = bool(config.get("allow_network", False))

    with tempfile.TemporaryDirectory(prefix="adh-verify-") as tmp:
        d = Path(tmp)
        (d / "solution.py").write_text(request.output, encoding="utf-8")
        (d / "test_candidate.py").write_text(
            "from solution import *  # noqa: F401,F403\n\n" + str(tests), encoding="utf-8"
        )
        if not allow_network:
            (d / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_candidate.py"]
        return _run(
            cmd,
            cwd=d,
            timeout_s=timeout_s,
            mem_mb=mem_mb,
            max_file_mb=max_file_mb,
            env=_sandbox_env(d, allow_network=allow_network),
        )


def run_script(request: VerifierRequest, config: dict) -> VerifierResult:
    """Alternative oracle: concatenate ``output`` + ``tests`` and run as a plain script.

    Passes iff the script exits 0 (use for assert-style checks with no pytest dep).
    Config keys are the same as :func:`run_pytest`.
    """
    tests = request.oracle("tests") or ""
    timeout_s = float(config.get("timeout_s", _DEFAULT_TIMEOUT_S))
    mem_mb = int(config.get("mem_mb", _DEFAULT_MEM_MB))
    max_file_mb = int(config.get("max_file_mb", _DEFAULT_MAX_FILE_MB))
    allow_network = bool(config.get("allow_network", False))

    with tempfile.TemporaryDirectory(prefix="adh-verify-") as tmp:
        d = Path(tmp)
        script = f"{request.output}\n\n{tests}\n"
        (d / "candidate.py").write_text(script, encoding="utf-8")
        if not allow_network:
            (d / "sitecustomize.py").write_text(_SITECUSTOMIZE, encoding="utf-8")
        return _run(
            [sys.executable, "candidate.py"],
            cwd=d,
            timeout_s=timeout_s,
            mem_mb=mem_mb,
            max_file_mb=max_file_mb,
            env=_sandbox_env(d, allow_network=allow_network),
        )


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    timeout_s: float,
    mem_mb: int,
    max_file_mb: int,
    env: dict[str, str],
) -> VerifierResult:
    cpu_seconds = max(1, int(timeout_s) + 1)
    kwargs: dict = {}
    preexec = _rlimits(cpu_seconds, mem_mb, max_file_mb)
    if preexec is not None:
        kwargs["preexec_fn"] = preexec
    if os.name == "posix":
        kwargs["process_group"] = 0  # own group so the timeout kill takes children too
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            **kwargs,
        )
    except OSError as exc:  # pragma: no cover - exec failure
        return VerifierResult(passed=False, detail=f"could not launch verifier subprocess: {exc}")
    try:
        stdout, stderr = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        _kill_group(proc)
        proc.communicate()  # reap; pipes close with the group
        return VerifierResult(passed=False, detail=f"timed out after {timeout_s:.0f}s")
    passed = proc.returncode == 0
    detail = tail(stdout + stderr) or f"exit code {proc.returncode}"
    return VerifierResult(passed=passed, detail=detail)
