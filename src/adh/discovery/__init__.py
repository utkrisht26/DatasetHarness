"""Phase 1.0 — the Source Discovery Agent and the approval gate."""

from __future__ import annotations

from .agent import DiscoveryConfig, SourceDiscoveryAgent
from .approval import (
    ApprovalDecision,
    Approver,
    AutoApprover,
    CallbackApprover,
    ThresholdApprover,
    apply_decision,
    render_plan,
)
from .models import (
    Aspect,
    CandidateSource,
    FetchPlan,
    RankedSource,
    SourceEvaluation,
)

__all__ = [
    "SourceDiscoveryAgent",
    "DiscoveryConfig",
    "Aspect",
    "CandidateSource",
    "SourceEvaluation",
    "RankedSource",
    "FetchPlan",
    "Approver",
    "ApprovalDecision",
    "AutoApprover",
    "CallbackApprover",
    "ThresholdApprover",
    "apply_decision",
    "render_plan",
]
