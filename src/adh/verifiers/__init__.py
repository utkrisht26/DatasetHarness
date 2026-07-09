"""Deterministic Tier-1 verifiers (OUTLINE §2.3 step 5, PHASE0-CONTRACT §0.5).

The oracle layer for hard-verifiable tasks: a record schema's ``x-verifier`` hook
names a ``module:function`` here, and :func:`run_verifier` runs it as a binary
keep/reject gate. Bundled oracles:

* ``adh.verifiers.code:run_pytest`` — candidate code vs unit tests in a sandbox
* ``adh.verifiers.math:check_equal`` — symbolic equality vs a gold answer (sympy)
* ``adh.verifiers.schema:validate_json`` — structured-output / constraint validation
"""

from __future__ import annotations

from .base import VerifierFn, VerifierRequest, resolve_verifier, run_verifier, tail

__all__ = [
    "VerifierRequest",
    "VerifierFn",
    "resolve_verifier",
    "run_verifier",
    "tail",
]
