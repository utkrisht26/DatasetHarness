"""Load a human-authored Job Spec YAML into the machine contract (§0.3.1 -> §0.3.2)."""

from __future__ import annotations

from pathlib import Path

import yaml

from .job_spec import JobSpec


def load_job_spec(path: str | Path) -> JobSpec:
    """Parse a Job Spec YAML file and validate it against the contract.

    Raises ``pydantic.ValidationError`` on any structural / cross-field violation
    (fail-fast at submission, per §0.3).
    """
    p = Path(path)
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"job spec at {p} must be a YAML mapping, got {type(data).__name__}")
    return JobSpec.model_validate(data)


def load_job_spec_from_str(text: str) -> JobSpec:
    """Parse a Job Spec from a YAML string (convenience for tests / APIs)."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("job spec text must parse to a YAML mapping")
    return JobSpec.model_validate(data)
