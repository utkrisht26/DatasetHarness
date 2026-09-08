"""Deterministic verifier plumbing (PHASE0-CONTRACT §0.5, OUTLINE §2.3 step 5).

A Tier-1 (hard) task carries an ``x-verifier`` hook on its record schema naming a
``module:function`` the harness imports and runs as the correctness oracle:

    { "kind": "unit_tests", "entry": "adh.verifiers.code:run_pytest", "timeout_s": 10 }

Every verifier is a plain function ``(request, config) -> VerifierResult`` where
``request`` carries the candidate payload (+ the gold/tests it is checked against)
and ``config`` is the ``x-verifier`` dict verbatim. Verifiers are treated as
**adversarially gameable** (OUTLINE risk table), so they must *reject* on any doubt
and never raise into the pipeline: :func:`run_verifier` converts a load/exec failure
into ``VerifierResult(passed=False, detail=...)`` rather than a crash.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

from adh.contract import VerifierResult


@dataclass(frozen=True)
class VerifierRequest:
    """One candidate presented to a deterministic verifier.

    ``output`` is the content under test; ``reference`` (a gold answer) and ``tests``
    (a unit-test source) are the oracle inputs a generator attaches to the candidate
    and that never ship in the final :class:`~adh.contract.Record`.
    """

    input: str
    output: str
    reference: str | None = None
    tests: str | None = None
    extras: dict = field(default_factory=dict)

    def oracle(self, key: str) -> object | None:
        """Read an oracle input, preferring a top-level field then ``extras``."""
        return getattr(self, key, None) or self.extras.get(key)


# A verifier: (candidate, x-verifier config) -> pass/fail verdict.
VerifierFn = Callable[[VerifierRequest, dict], VerifierResult]


def resolve_verifier(entry: str) -> VerifierFn:
    """Import a ``module:function`` verifier entry point.

    Raises ``ValueError`` on a malformed entry and ``ImportError``/``AttributeError``
    if the target does not exist — :func:`run_verifier` catches these.
    """
    module_name, sep, func_name = entry.partition(":")
    if not sep or not module_name or not func_name:
        raise ValueError(f"x-verifier entry must be 'module:function', got {entry!r}")
    module = importlib.import_module(module_name)
    fn = getattr(module, func_name)
    if not callable(fn):
        raise TypeError(f"x-verifier entry {entry!r} is not callable")
    return fn


def run_verifier(hook: dict, request: VerifierRequest) -> VerifierResult:
    """Resolve and run the ``x-verifier`` ``hook`` against ``request``, never raising.

    Any failure to load or execute the verifier is a *rejection* with the reason
    attached — a broken oracle must not let an unverified record through.
    """
    entry = hook.get("entry")
    if not entry:
        return VerifierResult(passed=False, detail="x-verifier hook is missing 'entry'")
    try:
        fn = resolve_verifier(entry)
    except Exception as exc:  # noqa: BLE001 - a missing/broken oracle rejects, never crashes
        return VerifierResult(passed=False, detail=f"could not load verifier {entry!r}: {exc}")
    try:
        result = fn(request, hook)
    except Exception as exc:  # noqa: BLE001 - a verifier that throws is a reject, with the reason
        return VerifierResult(passed=False, detail=f"verifier {entry!r} errored: {exc}")
    if not isinstance(result, VerifierResult):
        return VerifierResult(
            passed=False,
            detail=f"verifier {entry!r} returned {type(result).__name__}, not VerifierResult",
        )
    return result


def tail(text: str, *, limit: int = 800) -> str:
    """Trim verifier stdout/stderr to a bounded, single-block detail string."""
    text = (text or "").strip()
    return text if len(text) <= limit else "…" + text[-limit:]
