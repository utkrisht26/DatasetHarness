"""Verification verdict (OUTLINE §2.3).

One :class:`Verdict` per candidate records *why* it was kept or dropped and by which
stage — so the accept rate, per-stage drop counts, and judge agreement are all
observable, and a kept record carries its verifier result + judge scores straight
into the final :class:`~adh.contract.Record`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from adh.contract import VerifierResult

# The cheapest-first stages a verdict can be decided at (OUTLINE §2.3 ordering).
STAGE_QUALITY = "quality"
STAGE_VERIFIER = "verifier"
STAGE_JURY = "jury"
STAGE_PASSED = "passed"


@dataclass(frozen=True)
class Verdict:
    kept: bool
    stage: str
    reason: str = ""
    verifier_result: VerifierResult | None = None
    judge_scores: dict[str, float] | None = None
    judge_models: tuple[str, ...] = field(default_factory=tuple)

    @property
    def dropped(self) -> bool:
        return not self.kept

    def to_dict(self) -> dict:
        return {
            "kept": self.kept,
            "stage": self.stage,
            "reason": self.reason,
            "verifier_result": self.verifier_result.model_dump() if self.verifier_result else None,
            "judge_scores": self.judge_scores,
            "judge_models": list(self.judge_models),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Verdict":
        vr = data.get("verifier_result")
        return cls(
            kept=bool(data["kept"]),
            stage=data.get("stage", ""),
            reason=data.get("reason", ""),
            verifier_result=VerifierResult(**vr) if vr else None,
            judge_scores=data.get("judge_scores"),
            judge_models=tuple(data.get("judge_models", [])),
        )
