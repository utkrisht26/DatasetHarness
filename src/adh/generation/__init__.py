"""Engine B generation — seed/taxonomy engine (2.1) + record generation (2.2)."""

from __future__ import annotations

from .generator import (
    Generator,
    HeuristicGenerator,
    LlmGenerator,
    build_generator,
)
from .media_packager import MediaPackager
from .models import (
    DEFAULT_DIFFICULTIES,
    DEFAULT_PERSONAS,
    Candidate,
    SeedPoint,
    make_candidate_id,
)
from .seed import (
    DEFAULT_DIVERSITY_STRATEGY,
    EVOLUTION_OPS,
    DiversityStrategy,
    EvolInstructStrategy,
    NoveltyGate,
    PersonaGridStrategy,
    PersonaHubStrategy,
    SeedEngine,
    UnitContext,
    CoverageReport,
    coverage_report,
    persona_grid,
    resolve_strategy,
)

__all__ = [
    # models
    "SeedPoint",
    "Candidate",
    "make_candidate_id",
    "MediaPackager",
    "DEFAULT_PERSONAS",
    "DEFAULT_DIFFICULTIES",
    # diversity strategies (JobSpec.diversity_strategy)
    "DEFAULT_DIVERSITY_STRATEGY",
    "EVOLUTION_OPS",
    "DiversityStrategy",
    "EvolInstructStrategy",
    "PersonaGridStrategy",
    "PersonaHubStrategy",
    "UnitContext",
    "persona_grid",
    "resolve_strategy",
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
