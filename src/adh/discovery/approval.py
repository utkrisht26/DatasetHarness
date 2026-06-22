"""Source approval gate (OUTLINE §1.0, default ON).

Discovery presents a ranked source list + reasoning; a human (or an autonomous
policy) approves a subset before anything is fetched. ``discovery.approval_gate=false``
runs autonomous via :class:`AutoApprover`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

from .models import FetchPlan, RankedSource


@dataclass(frozen=True)
class ApprovalDecision:
    """Which sources were approved, and whether synthesis aspects proceed."""

    approved_source_ids: frozenset[str]
    approve_synthesis: bool = True
    reason: str = ""

    def approves(self, source: RankedSource) -> bool:
        return source.source_id in self.approved_source_ids


@runtime_checkable
class Approver(Protocol):
    def review(self, plan: FetchPlan) -> ApprovalDecision:
        """Decide which of the plan's sources to fetch."""
        ...


@dataclass
class AutoApprover:
    """Approve everything — used when ``approval_gate=false`` (autonomous runs)."""

    def review(self, plan: FetchPlan) -> ApprovalDecision:
        return ApprovalDecision(
            approved_source_ids=frozenset(s.source_id for s in plan.sources),
            approve_synthesis=True,
            reason="autonomous (approval_gate disabled)",
        )


@dataclass
class CallbackApprover:
    """Delegate the decision to a callable (CLI prompt, web UI, test stub)."""

    callback: Callable[[FetchPlan], ApprovalDecision]

    def review(self, plan: FetchPlan) -> ApprovalDecision:
        return self.callback(plan)


@dataclass
class ThresholdApprover:
    """Non-interactive policy: approve sources scoring at/above a threshold.

    Useful for batch/headless runs that still want a quality gate without a human.
    """

    min_score: float = 0.3
    require_license_ok: bool = True
    approve_synthesis: bool = True

    def review(self, plan: FetchPlan) -> ApprovalDecision:
        approved = {
            s.source_id
            for s in plan.sources
            if s.evaluation.score >= self.min_score
            and (s.evaluation.license_ok or not self.require_license_ok)
        }
        return ApprovalDecision(
            approved_source_ids=frozenset(approved),
            approve_synthesis=self.approve_synthesis,
            reason=f"threshold>={self.min_score} license_ok={self.require_license_ok}",
        )


def apply_decision(plan: FetchPlan, decision: ApprovalDecision) -> FetchPlan:
    """Return a new plan containing only the approved sources / synthesis aspects."""
    sources = tuple(s for s in plan.sources if decision.approves(s))
    synthesis = plan.synthesis_aspects if decision.approve_synthesis else ()
    notes = list(plan.notes) + [
        f"approval: {len(sources)}/{len(plan.sources)} sources, "
        f"synthesis={'on' if decision.approve_synthesis else 'off'} ({decision.reason})"
    ]
    return FetchPlan(
        job_id=plan.job_id,
        sources=sources,
        synthesis_aspects=synthesis,
        aspects=plan.aspects,
        notes=notes,
    )


def render_plan(plan: FetchPlan, *, limit: int = 50) -> str:
    """Human-readable ranked source list for the approval prompt / logs."""
    lines = [
        f"Fetch plan for job {plan.job_id!r}:",
        f"  {len(plan.sources)} ranked source(s), "
        f"{len(plan.synthesis_aspects)} synthesis aspect(s)",
    ]
    for i, s in enumerate(plan.sources[:limit], 1):
        e = s.evaluation
        flags = []
        if not e.license_ok:
            flags.append("LICENSE?")
        if not e.freshness_ok:
            flags.append("STALE?")
        flag_str = (" [" + ",".join(flags) + "]") if flags else ""
        lines.append(
            f"  {i:>2}. [{s.channel.value:<9}] score={e.score:<6} "
            f"shape={e.shape_match:<5} yield~{e.estimated_yield:<4} "
            f"lic={s.candidate.license or 'unknown'}{flag_str}\n"
            f"      {s.candidate.title}  <{s.candidate.locator}>"
        )
    for a in plan.synthesis_aspects:
        lines.append(f"  [synthesis] {a.topic}  (no external source — generated)")
    if plan.notes:
        lines.append("  notes:")
        lines.extend(f"    - {n}" for n in plan.notes)
    return "\n".join(lines)
