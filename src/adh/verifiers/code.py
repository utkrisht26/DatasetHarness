"""Tier-1 code verifier — run candidate code against unit tests in a sandbox.

This is the oracle the locked contract already names
(``adh.verifiers.code:run_pytest`` in ``schemas/code_record.schema.json``). It runs
the candidate ``output`` together with the candidate's ``tests`` in an isolated
subprocess with a wall-clock timeout, a CPU-time ``RLIMIT`` backstop, and its own
process group (killed as a group on timeout so no child survives).

Sandbox honesty: a subprocess with resource limits is a *floor*, not a jail — it
bounds runaway CPU/time but does not block filesystem or network. Production should
run this inside a container / nsjail / seccomp profile; the entry point and calling
convention stay identical, so that hardening is a deployment concern, not a code
change.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from adh.contract import VerifierResult

from .base import VerifierRequest, tail

_DEFAULT_TIMEOUT_S = 10.0


def _rlimits(cpu_seconds: int, mem_mb: int):
    """POSIX ``preexec_fn`` capping CPU time (and optionally address space)."""
    try:
        import resource
    except ImportError:  # pragma: no cover - non-POSIX
        return None

    def _apply() -> None:
        os.setsid()  # own process group so a timeout kill takes children too
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

    return _apply


def run_pytest(request: VerifierRequest, config: dict) -> VerifierResult:
    """Execute ``request.tests`` against ``request.output`` with pytest; pass iff green.

    The solution is written to ``solution.py`` and the tests to a test module that
    first ``from solution import *`` — so tests reference the solution's names
    directly. Config keys: ``timeout_s`` (wall-clock), ``mem_mb`` (0 = no AS cap).
    """
    tests = request.oracle("tests")
    if not tests:
        return VerifierResult(passed=False, detail="no unit tests supplied for code verification")

    timeout_s = float(config.get("timeout_s", _DEFAULT_TIMEOUT_S))
    mem_mb = int(config.get("mem_mb", 0))

    with tempfile.TemporaryDirectory(prefix="adh-verify-") as tmp:
        d = Path(tmp)
        (d / "solution.py").write_text(request.output, encoding="utf-8")
        (d / "test_candidate.py").write_text(
            "from solution import *  # noqa: F401,F403\n\n" + str(tests), encoding="utf-8"
        )
        cmd = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "test_candidate.py"]
        return _run(cmd, cwd=d, timeout_s=timeout_s, mem_mb=mem_mb)


def run_script(request: VerifierRequest, config: dict) -> VerifierResult:
    """Alternative oracle: concatenate ``output`` + ``tests`` and run as a plain script.

    Passes iff the script exits 0 (use for assert-style checks with no pytest dep).
    """
    tests = request.oracle("tests") or ""
    timeout_s = float(config.get("timeout_s", _DEFAULT_TIMEOUT_S))
    mem_mb = int(config.get("mem_mb", 0))
    with tempfile.TemporaryDirectory(prefix="adh-verify-") as tmp:
        d = Path(tmp)
        script = f"{request.output}\n\n{tests}\n"
        (d / "candidate.py").write_text(script, encoding="utf-8")
        return _run([sys.executable, "candidate.py"], cwd=d, timeout_s=timeout_s, mem_mb=mem_mb)


def _run(cmd: list[str], *, cwd: Path, timeout_s: float, mem_mb: int) -> VerifierResult:
    cpu_seconds = max(1, int(timeout_s) + 1)
    kwargs: dict = {}
    preexec = _rlimits(cpu_seconds, mem_mb)
    if preexec is not None:
        kwargs["preexec_fn"] = preexec
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0"},
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        return VerifierResult(passed=False, detail=f"timed out after {timeout_s:.0f}s")
    except OSError as exc:  # pragma: no cover - exec failure
        return VerifierResult(passed=False, detail=f"could not launch verifier subprocess: {exc}")
    passed = proc.returncode == 0
    detail = tail(proc.stdout + proc.stderr) or f"exit code {proc.returncode}"
    return VerifierResult(passed=passed, detail=detail)
