"""Engine B verification (OUTLINE §2.3) — tiered, cheapest-first, adversarially aware."""

from __future__ import annotations

from .jury import (
    DEFAULT_DIMENSIONS,
    DEFAULT_RUBRIC,
    HeuristicJudge,
    JudgePanel,
    LlmJuryPanel,
    PanelResult,
    build_judge,
)
from .models import (
    STAGE_JURY,
    STAGE_PASSED,
    STAGE_QUALITY,
    STAGE_VERIFIER,
    Verdict,
)
from .quality import QualityFilter, QualityResult
from .router import VerificationRouter, build_router

__all__ = [
    "Verdict",
    "STAGE_QUALITY",
    "STAGE_VERIFIER",
    "STAGE_JURY",
    "STAGE_PASSED",
    "QualityFilter",
    "QualityResult",
    "JudgePanel",
    "HeuristicJudge",
    "LlmJuryPanel",
    "PanelResult",
    "build_judge",
    "DEFAULT_DIMENSIONS",
    "DEFAULT_RUBRIC",
    "VerificationRouter",
    "build_router",
]
