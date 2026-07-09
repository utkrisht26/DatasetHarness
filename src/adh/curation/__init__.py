"""Engine B curation — near-dedup (2.4), decontamination (2.4), format/validate (2.5)."""

from __future__ import annotations

from .decontam import Decontaminator
from .dedup import DedupAssignment, Deduper, dedupe
from .format import assign_split, build_record, dead_letter_row

__all__ = [
    "Deduper",
    "DedupAssignment",
    "dedupe",
    "Decontaminator",
    "assign_split",
    "build_record",
    "dead_letter_row",
]
