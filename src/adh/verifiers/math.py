"""Tier-1 math verifier — symbolic equality of the answer against a gold reference.

Deterministic correctness for math tasks (OUTLINE §2.3: "sympy / symbolic match").
The candidate ``output`` and the ``reference`` gold are parsed and compared with
``sympy.simplify(a - b) == 0``, which decides equality up to algebra (``1/2`` ==
``0.5`` == ``sin(pi/6)``). Extraction is deliberate: a free-text answer is reduced to
its final expression (``\\boxed{...}``, ``answer =``, or the last numeric/expression
token) before parsing, and a parse failure falls back to normalized string equality
rather than silently passing.

sympy is the ``verify`` extra; it is imported lazily so the core stays importable.
"""

from __future__ import annotations

import re

from adh.contract import VerifierResult

from .base import VerifierRequest

_BOXED = re.compile(r"\\boxed\{([^}]*)\}")
_ANSWER = re.compile(r"(?:answer|result)\s*(?:is|:|=)\s*(.+)", re.IGNORECASE)
_FINAL_EXPR = re.compile(r"[-+]?[\d.,/*^()a-zA-Z_\s]+$")


def extract_answer(text: str) -> str:
    """Reduce a free-text solution to its final answer expression."""
    text = (text or "").strip()
    boxed = _BOXED.findall(text)
    if boxed:
        return boxed[-1].strip()
    last_line = text.splitlines()[-1].strip() if text else ""
    m = _ANSWER.search(text)
    if m:
        return m.group(1).strip().rstrip(".")
    return last_line.rstrip(".") or text


def _to_expr(raw: str):
    from sympy.parsing.sympy_parser import (
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    cleaned = raw.strip().replace("^", "**").replace(",", "").rstrip(".")
    transformations = standard_transformations + (implicit_multiplication_application,)
    return parse_expr(cleaned, transformations=transformations, evaluate=True)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", "", (text or "").strip().lower().rstrip("."))


def check_equal(request: VerifierRequest, config: dict) -> VerifierResult:
    """Pass iff the candidate answer is symbolically equal to the gold reference."""
    reference = request.oracle("reference")
    if reference is None:
        return VerifierResult(passed=False, detail="no reference answer for math verification")

    cand_raw = extract_answer(request.output) if config.get("extract", True) else request.output
    ref_raw = extract_answer(str(reference)) if config.get("extract", True) else str(reference)

    try:
        import sympy
    except ImportError:  # pragma: no cover - needs the verify extra
        equal = _normalize(cand_raw) == _normalize(ref_raw)
        return VerifierResult(passed=equal, detail=f"sympy unavailable; string match={equal}")

    try:
        diff = sympy.simplify(_to_expr(cand_raw) - _to_expr(ref_raw))
        equal = diff == 0
        return VerifierResult(
            passed=bool(equal), detail=f"symbolic: {cand_raw!r} vs {ref_raw!r} -> equal={bool(equal)}"
        )
    except Exception as exc:  # noqa: BLE001 - unparseable expr falls back to string equality
        equal = _normalize(cand_raw) == _normalize(ref_raw)
        return VerifierResult(
            passed=equal, detail=f"symbolic parse failed ({exc}); string match={equal}"
        )
