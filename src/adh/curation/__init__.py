"""Engine B curation — near-dedup (2.4), decontamination + PII policy (2.4), format/validate (2.5)."""

from __future__ import annotations

from .decontam import Decontaminator
from .dedup import DedupAssignment, Deduper, dedupe
from .format import assign_split, build_record, dead_letter_row
from .pii import ConfigurationError, PIIEntity, PIIRedactor, PresidioRedactor, resolve_redactor

__all__ = [
    "Deduper",
    "DedupAssignment",
    "dedupe",
    "Decontaminator",
    "assign_split",
    "build_record",
    "dead_letter_row",
    "ConfigurationError",
    "PIIEntity",
    "PIIRedactor",
    "PresidioRedactor",
    "resolve_redactor",
]
