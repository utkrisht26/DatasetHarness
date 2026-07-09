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
    result = validate_json(
        VerifierRequest(input="", output='{"name": "ada"}'), {"schema": schema}
    )
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
    result = run_verifier({"entry": "adh.verifiers.nope:missing"}, VerifierRequest(input="", output=""))
    assert isinstance(result, VerifierResult)
    assert result.passed is False
    assert "could not load" in (result.detail or "")


def test_run_verifier_rejects_missing_entry():
    assert run_verifier({}, VerifierRequest(input="", output="")).passed is False


def test_resolve_verifier_rejects_malformed_entry():
    with pytest.raises(ValueError):
        resolve_verifier("no-colon-here")
