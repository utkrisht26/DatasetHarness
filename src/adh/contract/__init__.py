"""Phase 0 contract (LOCKED) as code.

The Job Spec, the Record Schema, the verifiability tiers, and the validation
strategy from ``PHASE0-CONTRACT.md``. Everything downstream is built against these.
"""

from __future__ import annotations

from .enums import DiscoveryMode, SourceChannel, VerifiabilityTier
from .job_spec import (
    DedupPolicy,
    Discovery,
    DiscoveryConstraints,
    JobSpec,
    JudgeConfig,
    Limits,
    TargetSize,
)
from .loader import load_job_spec, load_job_spec_from_str
from .record import Record, VerifierResult
from .validation import (
    JobSpecError,
    RecordValidationError,
    ValidationReport,
    check_job_spec,
    get_verifier_hook,
    load_record_schema,
    validate_record,
)

__all__ = [
    # enums
    "VerifiabilityTier",
    "DiscoveryMode",
    "SourceChannel",
    # job spec
    "JobSpec",
    "Discovery",
    "DiscoveryConstraints",
    "JudgeConfig",
    "Limits",
    "DedupPolicy",
    "TargetSize",
    # record
    "Record",
    "VerifierResult",
    # loader
    "load_job_spec",
    "load_job_spec_from_str",
    # validation
    "ValidationReport",
    "JobSpecError",
    "RecordValidationError",
    "check_job_spec",
    "validate_record",
    "load_record_schema",
    "get_verifier_hook",
]
