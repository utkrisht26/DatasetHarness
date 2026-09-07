"""Phase 0 contract (LOCKED) as code.

The Job Spec, the Record Schema, the verifiability tiers, and the validation
strategy from ``PHASE0-CONTRACT.md``. Everything downstream is built against these.
"""

from __future__ import annotations

from .enums import (
    MEDIA_MODALITIES,
    DiscoveryMode,
    SourceChannel,
    VerifiabilityTier,
    is_media_modality,
)
from .job_spec import (
    DedupPolicy,
    Discovery,
    DiscoveryConstraints,
    JobSpec,
    JudgeConfig,
    Limits,
    TargetSize,
    resolved_media_types,
)
from .loader import load_job_spec, load_job_spec_from_str
from .record import MediaRef, Record, VerifierResult
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
    "MEDIA_MODALITIES",
    "is_media_modality",
    # job spec
    "JobSpec",
    "Discovery",
    "DiscoveryConstraints",
    "JudgeConfig",
    "Limits",
    "DedupPolicy",
    "TargetSize",
    "resolved_media_types",
    # record
    "Record",
    "MediaRef",
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
