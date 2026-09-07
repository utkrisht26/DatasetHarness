"""Tiered verification router (OUTLINE §2.3, PHASE0-CONTRACT §0.2).

Selects the verification sub-graph from the job's ``verifiability_tier`` and runs it
cheapest-first — reject early, escalate only survivors:

* **Tier 1 (hard):** quality filter → deterministic verifier gate. Binary, no jury.
* **Tier 2 (soft):** quality filter → LLM-jury (mean rubric score ≥ ``min_score``).
* **Tier 3 (hybrid):** quality filter → deterministic gate → jury on the survivors
  (the graded dimension only). The gate slashes judge cost and removes
  confidently-wrong items the jury would mis-grade.

The deterministic gate reuses the record schema's ``x-verifier`` hook; the jury is any
:class:`~adh.verification.jury.JudgePanel`, called reference-guided when the candidate
carries a gold answer. Both are injected, so the whole router runs offline with the
heuristic judge and a real oracle for Tier-1.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from adh.contract import JobSpec, VerifiabilityTier
from adh.generation import Candidate
from adh.verifiers import run_verifier

from .jury import DEFAULT_DIMENSIONS, DEFAULT_RUBRIC, HeuristicJudge, JudgePanel
from .models import STAGE_JURY, STAGE_QUALITY, STAGE_VERIFIER, Verdict
from .quality import QualityFilter

logger = logging.getLogger(__name__)


@dataclass
class VerificationRouter:
    """Runs the per-tier cheapest-first verification stack over one candidate."""

    spec: JobSpec
    verifier_hook: dict | None = None  # x-verifier from the record schema (Tier 1/3)
    judge: JudgePanel | None = None
    quality: QualityFilter = field(default_factory=QualityFilter)
    dimensions: tuple[str, ...] = DEFAULT_DIMENSIONS
    rubric: str = ""
    _gate_degraded: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        self.judge = self.judge or HeuristicJudge()
        if not self.rubric:
            self.rubric = self._load_rubric() or DEFAULT_RUBRIC

    def verify(self, candidate: Candidate) -> Verdict:
        # 0. structural sanity — an empty input is never a valid record
        if not candidate.input.strip():
            return Verdict(False, STAGE_QUALITY, "empty input")
        # 1. cheapest: heuristic quality filter on the output
        q = self.quality.check(candidate.output, modality=self.spec.modality)
        if not q.ok:
            return Verdict(False, STAGE_QUALITY, q.reason)

        tier = candidate.verifiability_tier
        if tier == VerifiabilityTier.HARD:
            return self._gate(candidate)
        if tier == VerifiabilityTier.HYBRID:
            if not self.verifier_hook:
                # The gate is a cost optimization for hybrid, not a requirement: when
                # the record schema carries no x-verifier (e.g. plain-text records),
                # degrade to jury-only rather than rejecting every candidate.
                if not self._gate_degraded:
                    self._gate_degraded = True
                    logger.warning(
                        "tier-3 verification has no x-verifier hook; skipping the "
                        "deterministic gate and grading with the jury alone"
                    )
                return self._jury(candidate)
            gate = self._gate(candidate)
            if gate.dropped:
                return gate
            return self._jury(candidate, verifier_result=gate.verifier_result)
        return self._jury(candidate)

    # ------------------------------------------------------------------ stages
    def _gate(self, candidate: Candidate) -> Verdict:
        if not self.verifier_hook:
            return Verdict(
                False, STAGE_VERIFIER, "tier requires an x-verifier hook, none configured"
            )
        result = run_verifier(self.verifier_hook, candidate.to_verifier_request())
        return Verdict(
            kept=result.passed,
            stage=STAGE_VERIFIER,
            reason=result.detail or ("verifier passed" if result.passed else "verifier rejected"),
            verifier_result=result,
        )

    def _jury(self, candidate: Candidate, *, verifier_result=None) -> Verdict:
        """Grade the candidate with the jury; reference-guided when a gold exists.

        ``candidate.reference`` (the Tier-1-style oracle input, ``None`` when absent)
        is threaded into the panel so judges grade against ground truth rather than
        vibes; the default-``None`` protocol keeps reference-free calls compatible.
        """
        min_score = self.spec.judge_config.min_score if self.spec.judge_config else 3.5
        panel = self.judge.evaluate(
            input=candidate.input,
            output=candidate.output,
            rubric=self.rubric,
            dimensions=self.dimensions,
            reference=candidate.reference,
        )
        kept = bool(panel.models) and panel.overall >= min_score
        reason = (
            f"jury overall {panel.overall} vs min {min_score}"
            if panel.models
            else "jury abstained (no judge produced a score)"
        )
        return Verdict(
            kept=kept,
            stage=STAGE_JURY,
            reason=reason,
            verifier_result=verifier_result,
            judge_scores=panel.scores or None,
            judge_models=panel.models,
        )

    def _load_rubric(self) -> str:
        cfg = self.spec.judge_config
        ref = getattr(cfg, "rubric_ref", None) if cfg else None
        if not ref:
            return ""
        try:
            return Path(ref).read_text(encoding="utf-8")
        except OSError:
            return ""


def build_router(
    spec: JobSpec,
    *,
    verifier_hook: dict | None = None,
    judge: JudgePanel | None = None,
) -> VerificationRouter:
    return VerificationRouter(spec=spec, verifier_hook=verifier_hook, judge=judge)
