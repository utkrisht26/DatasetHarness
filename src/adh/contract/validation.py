"""Validation strategy (PHASE0-CONTRACT.md §0.5).

Two layers:

1. **Job Spec, filesystem-dependent checks** — the structural invariants live in
   :class:`adh.contract.job_spec.JobSpec` itself; the checks that need to read the
   referenced JSON Schema (Tier-1 ``x-verifier`` hook) or eval files live here and
   return a :class:`ValidationReport` rather than raising, so a job can be inspected
   before submission.

2. **Record, boundary validation** — every emitted record is validated against the
   per-job JSON Schema referenced by ``record_schema_ref`` (``additionalProperties:
   false`` + explicit ``required``) AND the base :class:`Record` Pydantic model.
   Failures are quarantined to a dead-letter store, never silently dropped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import jsonschema
from pydantic import ValidationError

from .enums import VerifiabilityTier
from .job_spec import JobSpec
from .record import Record


@dataclass
class ValidationReport:
    """Non-raising result for pre-submission inspection."""

    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_errors(self) -> None:
        if self.errors:
            raise JobSpecError("; ".join(self.errors))


class JobSpecError(ValueError):
    """Raised when a Job Spec fails a filesystem-dependent contract check."""


class RecordValidationError(ValueError):
    """Raised at the emission boundary when a record violates the contract."""

    def __init__(self, message: str, *, record: Any, cause: Exception | None = None):
        super().__init__(message)
        self.record = record
        self.cause = cause


def load_record_schema(record_schema_ref: str, *, base_dir: Path | None = None) -> dict:
    """Load and parse the JSON Schema referenced by ``record_schema_ref``."""
    path = Path(record_schema_ref)
    if not path.is_absolute() and base_dir is not None:
        path = base_dir / path
    text = path.read_text(encoding="utf-8")
    return json.loads(text)


def get_verifier_hook(schema: dict) -> dict | None:
    """Extract the ``x-verifier`` extension naming the deterministic verifier (§0.5).

    Example: ``{"kind": "unit_tests", "entry": "verifiers.code:run_pytest", "timeout_s": 10}``.
    """
    hook = schema.get("x-verifier")
    return hook if isinstance(hook, dict) else None


def check_job_spec(spec: JobSpec, *, base_dir: Path | None = None) -> ValidationReport:
    """Run the filesystem-dependent contract checks on an already-constructed JobSpec.

    The pure structural invariants have already passed (JobSpec would not have
    constructed otherwise). This adds:

      * ``record_schema_ref`` resolves to a readable, valid JSON Schema.
      * Tier 1 => that schema declares a deterministic ``x-verifier`` hook.
      * ``decontaminate_against`` paths exist (warning only — they may be produced later).
    """
    report = ValidationReport()

    if urlsplit(spec.record_schema_ref).scheme in {"http", "https"}:
        report.errors.append(
            f"record_schema_ref={spec.record_schema_ref!r} must name a local JSON Schema "
            "file; remote schema URLs are not supported. For text input/output records, "
            "use './schemas/qa_record.schema.json' with verifiability_tier=2 and a "
            "judge_config, or provide a Tier-1 schema with an executable x-verifier."
        )
        return report

    try:
        schema = load_record_schema(spec.record_schema_ref, base_dir=base_dir)
    except FileNotFoundError:
        report.errors.append(f"record_schema_ref not found: {spec.record_schema_ref}")
        return report
    except (json.JSONDecodeError, OSError) as exc:
        report.errors.append(f"record_schema_ref is not readable JSON: {exc}")
        return report

    if not isinstance(schema, dict):
        report.errors.append("record_schema_ref must contain a JSON Schema object")
        return report

    try:
        jsonschema.Draft202012Validator.check_schema(schema)
    except jsonschema.SchemaError as exc:
        report.errors.append(f"record_schema_ref is not a valid JSON Schema: {exc.message}")
        return report

    if spec.verifiability_tier == VerifiabilityTier.HARD:
        hook = get_verifier_hook(schema)
        if hook is None:
            report.errors.append(
                "verifiability_tier=1 (hard) requires the record schema to declare an "
                "'x-verifier' hook naming a deterministic verifier (module:function). "
                "For descriptive text records, use verifiability_tier=2 with judge_config."
            )
        elif not hook.get("entry"):
            report.errors.append("x-verifier hook is missing required 'entry' (module:function)")

    if schema.get("additionalProperties", None) is not False:
        report.warnings.append(
            "record schema should set additionalProperties:false to reject unknown fields"
        )

    # The delivery layout must resolve to a registered builder. Function-local
    # import: adh.delivery imports this contract package, so the contract cannot
    # import it at module load time.
    from adh.delivery.layouts import known_layouts

    available = known_layouts()
    if spec.delivery_layout not in available:
        report.errors.append(
            f"delivery_layout={spec.delivery_layout!r} is not a registered layout "
            f"(available: {', '.join(available)})"
        )

    for ref in spec.decontaminate_against:
        p = Path(ref)
        if not p.is_absolute() and base_dir is not None:
            p = base_dir / p
        if not p.exists():
            report.warnings.append(
                f"decontaminate_against path does not exist yet: {ref} "
                "(must be present before any eval split is emitted)"
            )

    return report


def validate_record(
    record: Record | dict,
    *,
    schema: dict | None = None,
) -> Record:
    """Validate one record at the emission boundary.

    Checks the base :class:`Record` Pydantic model and, when given, the per-job JSON
    Schema. Raises :class:`RecordValidationError` (carrying the offending payload) on
    failure so callers can route it to the dead-letter store.
    """
    try:
        model = record if isinstance(record, Record) else Record.model_validate(record)
    except ValidationError as exc:
        raise RecordValidationError(
            f"record failed base Record validation: {exc.error_count()} error(s)",
            record=record,
            cause=exc,
        ) from exc

    if schema is not None:
        payload = model.model_dump(mode="json")
        try:
            jsonschema.validate(
                instance=payload, schema=schema, cls=jsonschema.Draft202012Validator
            )
        except jsonschema.ValidationError as exc:
            raise RecordValidationError(
                f"record failed per-job JSON Schema at {list(exc.absolute_path)}: {exc.message}",
                record=record,
                cause=exc,
            ) from exc

    return model
