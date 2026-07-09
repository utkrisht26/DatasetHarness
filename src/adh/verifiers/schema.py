"""Tier-1 schema/constraint verifier — the candidate output must be valid structured data.

Deterministic correctness for extraction / structured-output tasks (OUTLINE §2.3:
"schema/constraint validation"). The candidate ``output`` must parse as JSON and, when
the ``x-verifier`` config carries a ``schema``, validate against it (Draft 2020-12).
Optional ``required_keys`` gives a lightweight constraint check without a full schema.
"""

from __future__ import annotations

import json

import jsonschema

from adh.contract import VerifierResult

from .base import VerifierRequest


def validate_json(request: VerifierRequest, config: dict) -> VerifierResult:
    """Pass iff ``output`` is JSON that satisfies the configured schema / key set."""
    try:
        obj = json.loads(request.output)
    except (json.JSONDecodeError, TypeError) as exc:
        return VerifierResult(passed=False, detail=f"output is not valid JSON: {exc}")

    required_keys = config.get("required_keys")
    if required_keys:
        if not isinstance(obj, dict):
            return VerifierResult(passed=False, detail="output is not a JSON object")
        missing = [k for k in required_keys if k not in obj]
        if missing:
            return VerifierResult(passed=False, detail=f"missing required keys: {missing}")

    schema = config.get("schema")
    if schema is None:
        return VerifierResult(passed=True, detail="parsed as JSON (no schema constraint)")

    try:
        jsonschema.validate(instance=obj, schema=schema, cls=jsonschema.Draft202012Validator)
    except jsonschema.ValidationError as exc:
        return VerifierResult(
            passed=False, detail=f"schema violation at {list(exc.absolute_path)}: {exc.message}"
        )
    except jsonschema.SchemaError as exc:
        return VerifierResult(passed=False, detail=f"x-verifier schema is itself invalid: {exc.message}")
    return VerifierResult(passed=True, detail="valid JSON matching schema")
