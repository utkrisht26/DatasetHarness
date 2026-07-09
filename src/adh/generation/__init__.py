"""Engine B generation — seed/taxonomy engine (2.1) + record generation (2.2)."""

from __future__ import annotations

from .generator import (
    Generator,
    HeuristicGenerator,
    LlmGenerator,
    build_generator,
)
from .models import (
    DEFAULT_DIFFICULTIES,
    DEFAULT_PERSONAS,
    Candidate,
    SeedPoint,
    make_candidate_id,
)
from .seed import CoverageReport, NoveltyGate, SeedEngine, coverage_report

__all__ = [
    # models
    "SeedPoint",
    "Candidate",
    "make_candidate_id",
    "DEFAULT_PERSONAS",
    "DEFAULT_DIFFICULTIES",
    # seed engine
    "SeedEngine",
    "NoveltyGate",
    "CoverageReport",
    "coverage_report",
    # generation
    "Generator",
    "HeuristicGenerator",
    "LlmGenerator",
    "build_generator",
]
