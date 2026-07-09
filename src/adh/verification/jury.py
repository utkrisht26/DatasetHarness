"""LLM-as-jury (OUTLINE §2.3 step 8) — the Tier-2/3 graded oracle.

Soft-verifiable quality has no deterministic oracle, so it is graded by a jury with
the bias mitigations the contract mandates: a **panel of disjoint model families**
(PoLL — cheaper *and* less biased than one big judge), **length-controlled** rubric
scoring (judges are told not to reward verbosity), and reference-guided scoring where
a gold exists. The panel returns per-dimension scores; the router thresholds their
mean against ``judge_config.min_score``.

Two implementations, same seam as everywhere else: :class:`HeuristicJudge` scores
deterministically offline; :class:`LlmJuryPanel` runs the real models and meters them.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from adh.llm import UsageTracker

logger = logging.getLogger(__name__)

DEFAULT_DIMENSIONS = ("helpfulness", "correctness", "clarity")
DEFAULT_RUBRIC = (
    "Rate the response on each dimension from 0 (unusable) to 5 (excellent). "
    "Judge only quality and correctness for the given input; do NOT reward length or "
    "verbosity. A concise correct answer must score at least as high as a long one."
)
_WORD_RE = re.compile(r"[a-z0-9]+")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class PanelResult:
    scores: dict[str, float]  # per-dimension mean across the panel
    models: tuple[str, ...]

    @property
    def overall(self) -> float:
        return round(sum(self.scores.values()) / len(self.scores), 3) if self.scores else 0.0


@runtime_checkable
class JudgePanel(Protocol):
    def evaluate(
        self, *, input: str, output: str, rubric: str, dimensions: tuple[str, ...]
    ) -> PanelResult:
        """Score one input/output pair across ``dimensions`` on a 0–5 scale."""
        ...


# --------------------------------------------------------------------- heuristic
@dataclass
class HeuristicJudge:
    """Deterministic offline jury — length-adequacy + input relevance + structure."""

    model_id: str = "heuristic:judge"

    def evaluate(
        self, *, input: str, output: str, rubric: str, dimensions: tuple[str, ...]
    ) -> PanelResult:
        in_words = set(_WORD_RE.findall(input.lower()))
        out_words = _WORD_RE.findall(output.lower())
        out_set = set(out_words)
        n = len(out_words)

        # relevance: how much of the input's vocabulary the answer engages with
        relevance = (len(in_words & out_set) / len(in_words)) if in_words else 0.5
        # length adequacy: a broad plateau (12–400 words), penalize extremes only
        if n < 4:
            length = 0.2
        elif n < 12:
            length = 0.6
        elif n <= 400:
            length = 1.0
        else:
            length = 0.7
        # clarity: lexical variety (not one word repeated)
        variety = (len(out_set) / n) if n else 0.0

        helpfulness = 5.0 * (0.6 * length + 0.4 * relevance)
        correctness = 5.0 * (0.5 * relevance + 0.5 * min(length + 0.2, 1.0))
        clarity = 5.0 * (0.5 * length + 0.5 * min(variety * 1.5, 1.0))
        raw = {"helpfulness": helpfulness, "correctness": correctness, "clarity": clarity}
        scores = {d: round(min(5.0, max(0.0, raw.get(d, helpfulness))), 3) for d in dimensions}
        return PanelResult(scores=scores, models=(self.model_id,))


# --------------------------------------------------------------------- llm panel
@dataclass
class LlmJuryPanel:
    """Live PoLL panel over ≥2 disjoint model families (the ``agents`` extra)."""

    models: tuple[str, ...]
    usage: UsageTracker | None = None
    _judges: dict = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        from adh.llm import build_chat_model

        for model_id in self.models:
            self._judges[model_id] = build_chat_model(model_id, temperature=0.0)

    def evaluate(
        self, *, input: str, output: str, rubric: str, dimensions: tuple[str, ...]
    ) -> PanelResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = _judge_prompt(input, output, rubric, dimensions)
        per_dim: dict[str, list[float]] = {d: [] for d in dimensions}
        used: list[str] = []
        for model_id, model in self._judges.items():
            try:
                response = model.invoke(
                    [SystemMessage(content=rubric or DEFAULT_RUBRIC), HumanMessage(content=prompt)]
                )
            except Exception as exc:  # noqa: BLE001 - a flaky judge drops out of the panel
                logger.warning("judge %s failed: %s", model_id, exc)
                continue
            if self.usage is not None:
                self.usage.add_message(response)
            scores = _parse_scores(_text(response), dimensions)
            if scores is None:
                continue
            used.append(model_id)
            for d in dimensions:
                per_dim[d].append(scores[d])
        if not used:  # whole panel failed -> abstain low so the record is dropped
            return PanelResult(scores={d: 0.0 for d in dimensions}, models=())
        mean = {d: round(sum(v) / len(v), 3) for d, v in per_dim.items() if v}
        return PanelResult(scores=mean, models=tuple(used))


def _judge_prompt(input: str, output: str, rubric: str, dimensions: tuple[str, ...]) -> str:
    keys = ", ".join(f'"{d}": <0-5>' for d in dimensions)
    return (
        f"INPUT:\n{input}\n\nRESPONSE:\n{output}\n\n"
        f"Score the RESPONSE. Return ONLY a JSON object: {{{keys}}}."
    )


def _text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content)
    return str(content)


def _parse_scores(text: str, dimensions: tuple[str, ...]) -> dict[str, float] | None:
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    out: dict[str, float] = {}
    for d in dimensions:
        try:
            out[d] = min(5.0, max(0.0, float(obj.get(d, 0))))
        except (TypeError, ValueError):
            out[d] = 0.0
    return out


def build_judge(spec, *, usage: UsageTracker | None = None) -> JudgePanel:
    """Live PoLL panel when judge models are configured + importable; else heuristic."""
    cfg = getattr(spec, "judge_config", None)
    models = tuple(cfg.models) if cfg and cfg.models else ()
    if not models:
        return HeuristicJudge()
    try:
        return LlmJuryPanel(models=models, usage=usage)
    except Exception as exc:  # noqa: BLE001 - no extra / no key -> deterministic fallback
        logger.warning("LlmJuryPanel unavailable (%s); using HeuristicJudge", exc)
        return HeuristicJudge()
