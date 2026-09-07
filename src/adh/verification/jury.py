"""LLM-as-jury (OUTLINE §2.3 step 8) — the Tier-2/3 graded oracle.

Soft-verifiable quality has no deterministic oracle, so it is graded by a jury with
the bias mitigations the contract mandates: a **panel of disjoint model families**
(PoLL — cheaper *and* less biased than one big judge), **length-controlled** rubric
scoring (judges are told not to reward verbosity), **position-swap** scoring (every
item is graded in both presentation orders and averaged — RESEARCH.md measures GPT-4
at ~65% order-consistency), and reference-guided scoring where a gold exists. The
panel returns per-dimension scores; the router thresholds their mean against
``judge_config.min_score``.

Implementations, same seam as everywhere else: :class:`HeuristicJudge` scores
deterministically offline; :class:`LlmJuryPanel` runs the real models and meters
them; :class:`SwapConsistencyPanel` wraps any panel to cancel position bias.
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
        self,
        *,
        input: str,
        output: str,
        rubric: str,
        dimensions: tuple[str, ...],
        reference: str | None = None,
    ) -> PanelResult:
        """Score one input/output pair across ``dimensions`` on a 0–5 scale.

        ``reference`` is the gold answer when the candidate carries one — judges are
        expected to grade correctness against it rather than vibes; ``None`` (the
        default) keeps reference-free calls protocol-compatible.
        """
        ...


# --------------------------------------------------------------------- heuristic
@dataclass
class HeuristicJudge:
    """Deterministic offline jury — length-adequacy + input relevance + structure."""

    model_id: str = "heuristic:judge"

    def evaluate(
        self,
        *,
        input: str,
        output: str,
        rubric: str,
        dimensions: tuple[str, ...],
        reference: str | None = None,
    ) -> PanelResult:
        """Deterministically score the output; ``reference`` is accepted but unused.

        The heuristic scorer is slot- and reference-agnostic by design — no position
        bias to swap away and no ground truth to consult — so accepting the kwarg is
        pure protocol compatibility and the scoring behavior never changes.
        """
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
        self,
        *,
        input: str,
        output: str,
        rubric: str,
        dimensions: tuple[str, ...],
        reference: str | None = None,
    ) -> PanelResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        prompt = _judge_prompt(input, output, rubric, dimensions, reference=reference)
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


def _judge_prompt(
    input: str,
    output: str,
    rubric: str,
    dimensions: tuple[str, ...],
    reference: str | None = None,
) -> str:
    keys = ", ".join(f'"{d}": <0-5>' for d in dimensions)
    reference_block = (
        f"\n\nREFERENCE (gold answer — grade the RESPONSE's correctness against it):\n{reference}"
        if reference
        else ""
    )
    return (
        f"INPUT:\n{input}\n\nRESPONSE:\n{output}{reference_block}\n\n"
        f"Score the RESPONSE. Return ONLY a JSON object: {{{keys}}}."
    )


# ------------------------------------------------------------------ position-swap
@dataclass(frozen=True)
class SwapConsistencyPanel:
    """Position-bias mitigation (PHASE0-CONTRACT §0.2 "position-swapped pairwise").

    Wraps any :class:`JudgePanel` and scores every item twice — once with the
    input/response pair presented in canonical order and once with the presentation
    order swapped — then returns the per-dimension mean. Averaging cancels a judge's
    systematic preference for whichever text occupies a given prompt slot; a judge
    that flatters one order and pans the other lands on the same middle score either
    way. A pass that abstains (empty ``models``) contributes nothing to the average;
    if both abstain the wrapper abstains too. The deterministic :class:`HeuristicJudge`
    has no slot preference, so ``build_judge`` never wraps it.
    """

    inner: JudgePanel

    def evaluate(
        self,
        *,
        input: str,
        output: str,
        rubric: str,
        dimensions: tuple[str, ...],
        reference: str | None = None,
    ) -> PanelResult:
        passes = [
            self.inner.evaluate(
                input=input, output=output, rubric=rubric, dimensions=dimensions, reference=reference
            ),
            self.inner.evaluate(
                input=output, output=input, rubric=rubric, dimensions=dimensions, reference=reference
            ),
        ]
        models: list[str] = []
        for p in passes:
            for m in p.models:
                if m not in models:
                    models.append(m)
        scored = [p for p in passes if p.models]
        if not scored:
            return PanelResult(scores={d: 0.0 for d in dimensions}, models=())
        mean = {
            d: round(sum(p.scores.get(d, 0.0) for p in scored) / len(scored), 3)
            for d in dimensions
        }
        return PanelResult(scores=mean, models=tuple(models))


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
    """Live PoLL panel when judge models are configured + importable; else heuristic.

    Model selection is configuration, not code (Engine-A parity): the
    ``ADH_JUDGE_MODELS`` deployment override wins when set, else the spec's
    ``judge_config.models`` panel applies. Honors ``judge_config.position_swap``
    (default True): the live panel is wrapped in :class:`SwapConsistencyPanel` so
    every item is scored in both presentation orders. The deterministic heuristic
    path is returned unwrapped — it has no presentation-order bias to mitigate.
    """
    cfg = getattr(spec, "judge_config", None)
    models = _configured_judge_models(cfg)
    swap = bool(getattr(cfg, "position_swap", False)) if cfg is not None else False
    panel: JudgePanel
    if not models:
        panel = HeuristicJudge()
    else:
        try:
            panel = LlmJuryPanel(models=models, usage=usage)
        except Exception as exc:  # noqa: BLE001 - no extra / no key -> deterministic fallback
            logger.warning("LlmJuryPanel unavailable (%s); using HeuristicJudge", exc)
            panel = HeuristicJudge()
    if swap and isinstance(panel, LlmJuryPanel):
        return SwapConsistencyPanel(panel)
    return panel


def _configured_judge_models(cfg) -> tuple[str, ...]:
    """Deployment panel (``ADH_JUDGE_MODELS``) first, else the spec's own."""
    from adh.contract.families import distinct_families
    from adh.llm import JUDGE_MODELS_ENV, env_model_list

    env_ids = env_model_list(JUDGE_MODELS_ENV)
    if not env_ids:
        return tuple(cfg.models) if cfg and cfg.models else ()
    families = distinct_families(env_ids)
    if len(families) < 2:
        logger.warning(
            "%s override lists <2 distinct model families (%s) — the PoLL "
            "disjoint-family guarantee is weakened for this deployment",
            JUDGE_MODELS_ENV,
            sorted(families),
        )
    logger.warning(
        "judge panel overridden by %s=%s (spec configured %s)",
        JUDGE_MODELS_ENV,
        ",".join(env_ids),
        list(cfg.models) if cfg and cfg.models else [],
    )
    return tuple(env_ids)
