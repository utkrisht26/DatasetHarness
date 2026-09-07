"""Verification upgrades (OUTLINE §2.3 steps 8-9), offline.

Covers the four adversarial-robustness additions:

* :class:`~adh.verification.jury.SwapConsistencyPanel` — position-swap pairwise
  jury: every item is scored twice with the presentation order swapped and the
  per-dimension scores averaged, cancelling a judge's systematic slot preference
  (RESEARCH.md: GPT-4 ~65% order-consistency). The heuristic judge is deterministic
  and needs no wrapping.
* Reference-guided scoring — a candidate's gold ``reference`` is threaded into the
  jury call so judges grade against ground truth rather than vibes.
* Paraphrase decontamination gate — token-set Jaccard + contiguous n-gram
  containment against held-out spans catches rephrased copies that miss the exact
  n-gram index.
* Negative-control verifier tests — known-bad candidates pushed through the FULL
  :class:`VerificationRouter` for Tier 1 and Tier 3 must be dropped; guards against
  verifier regressions that accept everything.

Everything here runs hermetically with fakes; no network, no API keys.
"""

from __future__ import annotations

import pytest

import adh.verification.jury as jury_module
from adh.contract import VerifiabilityTier, get_verifier_hook, load_record_schema
from adh.curation import Decontaminator
from adh.generation import Candidate
from adh.verification import (
    HeuristicJudge,
    VerificationRouter,
    build_router,
)
from adh.verification.jury import (
    JudgePanel,
    PanelResult,
    SwapConsistencyPanel,
    _judge_prompt,
    build_judge,
)

# ------------------------------------------------------------------ helpers


def _spec(tier=2, modality="text", **kw):
    from adh.contract import JobSpec

    base = dict(
        name="job",
        description="d",
        motivation="m",
        intended_uses=["supervised-fine-tuning"],
        verifiability_tier=tier,
        task_categories=["question-answering"],
        languages=["en"],
        modality=modality,
        target_size={"records": 10},
        record_schema_ref="schemas/qa_record.schema.json",
        allowed_licenses=["MIT"],
        generator_models=[],
        decontaminate_against=["./eval/heldout_rust_qa.jsonl"],
        output_license="MIT",
    )
    if tier in (2, 3):
        base["judge_config"] = {
            "models": ["anthropic:claude-sonnet-4-6", "openai:gpt-judge"],
            "min_score": 3.5,
        }
    base.update(kw)
    return JobSpec(**base)


def _cand(tier=2, *, input="Explain async macros clearly.", output="Async macros expand code at compile time to reduce boilerplate.", **kw):
    return Candidate(
        candidate_id="c1",
        job_id="job",
        aspect_id="a1",
        verifiability_tier=VerifiabilityTier(tier),
        input=input,
        output=output,
        **kw,
    )


class RecordingJudge:
    """Returns scripted results in order and records every evaluate() call."""

    model_id = "fake:recording"

    def __init__(self, results):
        self.results = list(results)
        self.calls: list[dict] = []

    def evaluate(self, *, input, output, rubric, dimensions, reference=None):
        self.calls.append({"input": input, "output": output, "reference": reference})
        result = self.results.pop(0)
        if isinstance(result, PanelResult):
            return result
        return PanelResult(scores={d: result[d] for d in dimensions}, models=(self.model_id,))


def _scores(value: float) -> dict[str, float]:
    return {"helpfulness": value, "correctness": value, "clarity": value}


# ------------------------------------------------------------------ swap panel


def test_swap_panel_satisfies_judge_protocol():
    assert isinstance(SwapConsistencyPanel(HeuristicJudge()), JudgePanel)


def test_swap_panel_calls_inner_twice_with_swapped_presentation():
    judge = RecordingJudge([_scores(4.0), _scores(4.0)])
    panel = SwapConsistencyPanel(judge)
    dims = ("helpfulness", "correctness", "clarity")
    panel.evaluate(input="the question", output="the answer", rubric="r", dimensions=dims)
    assert len(judge.calls) == 2
    # canonical pass
    assert judge.calls[0]["input"] == "the question"
    assert judge.calls[0]["output"] == "the answer"
    # presentation-order-swapped pass
    assert judge.calls[1]["input"] == "the answer"
    assert judge.calls[1]["output"] == "the question"


def test_swap_panel_averages_scores_across_both_orders():
    judge = RecordingJudge([_scores(5.0), _scores(3.0)])
    dims = ("helpfulness", "correctness", "clarity")
    result = SwapConsistencyPanel(judge).evaluate(
        input="q", output="a", rubric="r", dimensions=dims
    )
    assert result.scores == {d: 4.0 for d in dims}
    assert result.models == ("fake:recording",)  # same model both passes -> deduped
    assert result.overall == pytest.approx(4.0)


def test_swap_panel_unions_models_from_both_passes():
    class TwoModelJudge(RecordingJudge):
        def __init__(self):
            super().__init__([_scores(4.0), _scores(2.0)])

        def evaluate(self, **kw):  # second pass reports a different model id
            self.calls.append(kw)
            self.results.pop(0)
            which = len(self.calls)
            return PanelResult(scores=_scores(4.0 if which == 1 else 2.0), models=(f"m{which}",))

    result = SwapConsistencyPanel(TwoModelJudge()).evaluate(
        input="q", output="a", rubric="r", dimensions=("helpfulness", "correctness", "clarity")
    )
    assert set(result.models) == {"m1", "m2"}
    assert result.overall == pytest.approx(3.0)


def test_swap_panel_reference_threaded_to_both_passes():
    judge = RecordingJudge([_scores(4.0), _scores(4.0)])
    dims = ("helpfulness", "correctness", "clarity")
    SwapConsistencyPanel(judge).evaluate(
        input="q", output="a", rubric="r", dimensions=dims, reference="gold"
    )
    assert judge.calls[0]["reference"] == "gold"
    assert judge.calls[1]["reference"] == "gold"


def test_swap_panel_survives_single_abstaining_pass():
    abstain = PanelResult(scores={d: 0.0 for d in ("helpfulness", "correctness", "clarity")}, models=())
    judge = RecordingJudge([abstain, _scores(4.0)])
    dims = ("helpfulness", "correctness", "clarity")
    result = SwapConsistencyPanel(judge).evaluate(input="q", output="a", rubric="r", dimensions=dims)
    assert result.models == ("fake:recording",)
    assert result.overall == pytest.approx(4.0)


def test_swap_panel_abstains_when_both_passes_fail():
    abstain = PanelResult(scores={d: 0.0 for d in ("helpfulness", "correctness", "clarity")}, models=())
    judge = RecordingJudge([abstain, abstain])
    dims = ("helpfulness", "correctness", "clarity")
    result = SwapConsistencyPanel(judge).evaluate(input="q", output="a", rubric="r", dimensions=dims)
    assert result.models == ()
    assert result.overall == 0.0


def test_position_bias_is_cancelled_by_swapping():
    """A judge with a systematic slot preference flatters one presentation order and
    pans the other; the swap wrapper averages both passes into an order-stable score."""

    class SlotBiasedJudge:
        """Flatters whichever text is longer sits in the input slot — pure slot bias."""

        model_id = "biased:slot"

        def evaluate(self, *, input, output, rubric, dimensions, reference=None):
            score = 5.0 if len(input) > len(output) else 1.0
            return PanelResult(scores={d: float(score) for d in dimensions}, models=(self.model_id,))

    dims = ("helpfulness", "correctness", "clarity")
    question, answer = "a much longer question statement", "the answer"
    biased = SlotBiasedJudge()
    raw = biased.evaluate(input=question, output=answer, rubric="r", dimensions=dims)
    assert raw.overall == 5.0  # flatters the canonical presentation

    wrapped = SwapConsistencyPanel(biased)
    canonical = wrapped.evaluate(input=question, output=answer, rubric="r", dimensions=dims)
    assert canonical.overall == pytest.approx(3.0)  # mean of the flattering + panning pass


# ------------------------------------------------------------------ reference-guided


def test_router_threads_candidate_reference_into_jury():
    judge = RecordingJudge([_scores(5.0)])
    router = build_router(_spec(tier=2), judge=judge)
    router.verify(_cand(tier=2, reference="42"))
    assert len(judge.calls) == 1
    assert judge.calls[0]["reference"] == "42"


def test_router_passes_none_reference_when_absent():
    judge = RecordingJudge([_scores(5.0)])
    router = build_router(_spec(tier=2), judge=judge)
    router.verify(_cand(tier=2))
    assert judge.calls[0]["reference"] is None


def test_heuristic_judge_accepts_and_ignores_reference():
    """The deterministic path must be unchanged by the new kwarg."""
    plain = HeuristicJudge().evaluate(
        input="q",
        output="an answer with several words in it",
        rubric="r",
        dimensions=("helpfulness", "correctness", "clarity"),
    )
    with_ref = HeuristicJudge().evaluate(
        input="q",
        output="an answer with several words in it",
        rubric="r",
        dimensions=("helpfulness", "correctness", "clarity"),
        reference="42",
    )
    assert plain.scores == with_ref.scores


def test_llm_prompt_includes_reference_block_only_when_present():
    without = _judge_prompt("q", "a", "rubric", ("helpfulness",))
    assert "REFERENCE" not in without
    with_ref = _judge_prompt("q", "a", "rubric", ("helpfulness",), reference="gold answer")
    assert "gold answer" in with_ref
    assert "REFERENCE" in with_ref


# ------------------------------------------------------------------ build_judge wiring


class FakeLivePanel:
    """Stands in for LlmJuryPanel so no live models / keys are needed."""

    def __init__(self, models, usage=None):
        self.models = tuple(models)
        self.usage = usage

    def evaluate(self, *, input, output, rubric, dimensions, reference=None):
        return PanelResult(scores=_scores(5.0), models=self.models)


def test_build_judge_wraps_live_panel_with_swap_by_default(monkeypatch):
    monkeypatch.setattr(jury_module, "LlmJuryPanel", FakeLivePanel)
    spec = _spec(tier=2)  # judge_config.position_swap defaults to True
    panel = build_judge(spec, usage=None)
    assert isinstance(panel, SwapConsistencyPanel)
    assert isinstance(panel.inner, FakeLivePanel)


def test_build_judge_respects_position_swap_false(monkeypatch):
    monkeypatch.setattr(jury_module, "LlmJuryPanel", FakeLivePanel)
    cfg = {"models": ["anthropic:claude-sonnet-4-6", "openai:gpt-judge"], "position_swap": False}
    panel = build_judge(_spec(tier=2, judge_config=cfg))
    assert isinstance(panel, FakeLivePanel)
    assert not isinstance(panel, SwapConsistencyPanel)


def test_build_judge_leaves_heuristic_path_unwrapped():
    """No live models -> deterministic HeuristicJudge, never wrapped."""
    panel = build_judge(_spec(tier=1))  # tier 1 has no judge_config at all
    assert isinstance(panel, HeuristicJudge)


# ------------------------------------------------------------------ env-configured panel
def test_build_judge_env_panel_overrides_the_spec(monkeypatch):
    """ADH_JUDGE_MODELS is the deployment-level redirect, Engine-A parity."""
    import adh.llm as llm

    monkeypatch.setattr(jury_module, "LlmJuryPanel", FakeLivePanel)
    monkeypatch.setenv(llm.models.JUDGE_MODELS_ENV, "openai:judge-a, anthropic:judge-b")

    panel = build_judge(_spec(tier=2))  # spec jury is anthropic+openai:...gpt-judge

    inner = panel.inner if isinstance(panel, SwapConsistencyPanel) else panel
    assert isinstance(inner, FakeLivePanel)
    assert inner.models == ("openai:judge-a", "anthropic:judge-b")


def test_build_judge_without_env_keeps_the_spec_panel(monkeypatch):
    import adh.llm as llm

    monkeypatch.setattr(jury_module, "LlmJuryPanel", FakeLivePanel)
    monkeypatch.delenv(llm.models.JUDGE_MODELS_ENV, raising=False)

    panel = build_judge(_spec(tier=2))

    inner = panel.inner if isinstance(panel, SwapConsistencyPanel) else panel
    assert inner.models == ("anthropic:claude-sonnet-4-6", "openai:gpt-judge")


def test_build_judge_env_override_beats_missing_config_too(monkeypatch):
    """A deployment panel applies even where a tier-1 spec configures none."""
    import adh.llm as llm

    monkeypatch.setattr(jury_module, "LlmJuryPanel", FakeLivePanel)
    monkeypatch.setenv(llm.models.JUDGE_MODELS_ENV, "openai:judge-a, anthropic:judge-b")

    panel = build_judge(_spec(tier=1))  # no judge_config at all

    assert isinstance(panel, FakeLivePanel)


def test_build_judge_single_family_override_warns(monkeypatch, caplog):
    """A one-family deployment panel silently weakens PoLL — say so loudly."""
    import logging

    import adh.llm as llm

    monkeypatch.setattr(jury_module, "LlmJuryPanel", FakeLivePanel)
    monkeypatch.setenv(llm.models.JUDGE_MODELS_ENV, "openai:solo-judge")
    with caplog.at_level(logging.WARNING, logger="adh.verification.jury"):
        build_judge(_spec(tier=2))
    assert any("famil" in record.message.lower() for record in caplog.records)


# ------------------------------------------------------------------ paraphrase decontam

HELDOUT = (
    '{"input": "How do async macros expand in Rust at compile time?",'
    ' "output": "Async macros expand code during compilation to remove boilerplate"}\n'
)
# Rephrased copy: one synonym swap ("async" -> "Asynchronous") lands inside every
# contiguous 13-gram window of the held-out span, so the exact-match index misses it
# entirely — but the token sets stay near-identical, so Jaccard catches it.
REPHRASED = (
    "How do async macros expand in Rust at compile time? Asynchronous macros expand "
    "code during compilation to remove boilerplate."
)
UNRELATED = "The quick brown fox jumps over the lazy dog near the river bank today."
NEAR_VERBATIM = (
    "Quick question: how do async macros expand in rust at compile time async macros "
    "expand code during compilation to remove boilerplate. Thanks!"
)


def _decon(tmp_path, body=HELDOUT, **kw):
    held = tmp_path / "heldout.jsonl"
    held.write_text(body, encoding="utf-8")
    return Decontaminator.from_paths([str(held)], ngram=13, **kw)


def test_paraphrase_gate_flags_rephrased_copy_exact_miss(tmp_path):
    d = _decon(tmp_path)
    assert d.active
    # exact-match index alone would MISS this rephrase...
    from adh.curation.decontam import _ngrams

    cand_grams = _ngrams(REPHRASED, 13)
    ref_grams = _ngrams(HELDOUT, 13)
    assert not (cand_grams & ref_grams), "fixture must not share an exact 13-gram"
    # ...but the paraphrase gate catches it
    assert d.is_contaminated(REPHRASED) is True


def test_paraphrase_gate_spares_unrelated_text(tmp_path):
    d = _decon(tmp_path)
    assert d.is_contaminated(UNRELATED) is False


def test_paraphrase_gate_thresholds_are_tunable(tmp_path):
    strict = _decon(tmp_path, paraphrase_jaccard=0.99, paraphrase_overlap=0.99)
    assert strict.is_contaminated(REPHRASED) is False
    loose = _decon(tmp_path, paraphrase_jaccard=0.2, paraphrase_overlap=0.05)
    assert loose.is_contaminated(REPHRASED) is True


def test_contamination_reason_names_the_gate(tmp_path):
    d = _decon(tmp_path)
    reason = d.contamination_reason(REPHRASED)
    assert reason is not None and "paraphrase" in reason
    assert d.contamination_reason(UNRELATED) is None


def test_short_candidate_below_fuzzy_floor_not_flagged(tmp_path):
    """Too little text for reliable fuzzy matching -> only the exact index decides."""
    d = _decon(tmp_path)
    assert d.is_contaminated("expand async macros boilerplate") is False


def test_exact_match_still_flagged_after_upgrade(tmp_path):
    d = _decon(tmp_path)
    assert d.is_contaminated(NEAR_VERBATIM) is True
    reason = d.contamination_reason(NEAR_VERBATIM)
    assert reason is not None and "exact" in reason  # caught by the deterministic layer


def test_decontaminator_from_spec_reads_dedup_policy(minimal_spec_dict):
    from adh.contract import JobSpec

    minimal_spec_dict["dedup_policy"] = {"method": "minhash", "ngram": 6, "jaccard": 0.55}
    spec = JobSpec(**minimal_spec_dict)
    d = Decontaminator.from_spec(spec)
    assert d.paraphrase_jaccard == 0.55
    assert d.paraphrase_ngram == 6


def test_decontaminator_from_spec_defaults_without_policy_fields(tmp_path):
    from adh.curation.decontam import (
        DEFAULT_PARAPHRASE_JACCARD,
        DEFAULT_PARAPHRASE_NGRAM,
        DEFAULT_PARAPHRASE_OVERLAP,
    )

    d = _decon(tmp_path)
    assert d.paraphrase_jaccard == DEFAULT_PARAPHRASE_JACCARD
    assert d.paraphrase_overlap == DEFAULT_PARAPHRASE_OVERLAP
    assert d.paraphrase_ngram == DEFAULT_PARAPHRASE_NGRAM


# ------------------------------------------------------------------ negative controls

CODE_HOOK_TIMEOUT = 30
MATH_HOOK = {"kind": "math", "entry": "adh.verifiers.math:check_equal", "timeout_s": CODE_HOOK_TIMEOUT}


class ExplodingJudge:
    """Any jury invocation on a gated-away candidate is a regression."""

    model_id = "fake:exploding"

    def evaluate(self, **_kw):
        raise AssertionError("jury must never run after the deterministic gate dropped the candidate")


GOOD_CODE = "def solve(x):\n    return x * 2\n"
BAD_CODE_CASES = [
    # every mutant must FAIL `assert solve(2) == 4` while looking plausible
    ("subtly-mutated-operator", "def solve(x):\n    return x * 3\n"),  # *2 mutated to *3
    ("off-by-one", "def solve(x):\n    return x * 2 + 1\n"),
    ("wrong-arithmetic", "def solve(x):\n    return x + x + x\n"),
]
CODE_TESTS = "def test_solve():\n    assert solve(2) == 4\n"


@pytest.mark.parametrize("case_name,bad_code", BAD_CODE_CASES, ids=[c[0] for c in BAD_CODE_CASES])
def test_negative_control_tier1_code_dropped(case_name, bad_code):
    schema = load_record_schema("schemas/code_record.schema.json")
    hook = {**get_verifier_hook(schema), "timeout_s": CODE_HOOK_TIMEOUT}
    router = VerificationRouter(spec=_spec(tier=1, modality="code"), verifier_hook=hook)
    cand = _cand(tier=1, input="Double the input.", output=bad_code, tests=CODE_TESTS)
    verdict = router.verify(cand)
    assert verdict.kept is False, f"{case_name}: known-bad code must not survive verification"


def test_positive_control_tier1_code_kept():
    schema = load_record_schema("schemas/code_record.schema.json")
    hook = {**get_verifier_hook(schema), "timeout_s": CODE_HOOK_TIMEOUT}
    router = VerificationRouter(spec=_spec(tier=1, modality="code"), verifier_hook=hook)
    cand = _cand(tier=1, input="Double the input.", output=GOOD_CODE, tests=CODE_TESTS)
    verdict = router.verify(cand)
    assert verdict.kept is True, "positive control must pass or the negative controls are vacuous"


MATH_BAD_CASES = [
    ("wrong-final-answer", "Step one: add. Therefore the answer is \\boxed{5}.", "4"),
    ("sign-flip", "The final answer is \\boxed{-4}.", "4"),
    ("arithmetic-slip", "So the result is \\boxed{404}.", "400"),
]


@pytest.mark.parametrize("case_name,output_text,reference", MATH_BAD_CASES, ids=[c[0] for c in MATH_BAD_CASES])
def test_negative_control_tier1_math_dropped(case_name, output_text, reference):
    router = VerificationRouter(spec=_spec(tier=1), verifier_hook=MATH_HOOK)
    cand = _cand(tier=1, input="What is 20 * 20?", output=output_text, reference=reference)
    verdict = router.verify(cand)
    assert verdict.kept is False, f"{case_name}: known-wrong final answer must be dropped"
    assert verdict.stage == "verifier"


def test_positive_control_tier1_math_kept():
    router = VerificationRouter(spec=_spec(tier=1), verifier_hook=MATH_HOOK)
    cand = _cand(tier=1, input="What is 20 * 20?", output="The answer is \\boxed{400}.", reference="4 * 100")
    assert router.verify(cand).kept is True


@pytest.mark.parametrize("case_name,bad_code", BAD_CODE_CASES, ids=[c[0] for c in BAD_CODE_CASES])
def test_negative_control_tier3_gate_drops_before_jury(case_name, bad_code):
    schema = load_record_schema("schemas/code_record.schema.json")
    hook = {**get_verifier_hook(schema), "timeout_s": CODE_HOOK_TIMEOUT}
    router = VerificationRouter(
        spec=_spec(tier=3, modality="code"),
        verifier_hook=hook,
        judge=ExplodingJudge(),  # fires if the jury ever sees this candidate
    )
    cand = _cand(tier=3, input="Double the input.", output=bad_code, tests=CODE_TESTS)
    verdict = router.verify(cand)
    assert verdict.kept is False
    assert verdict.stage == "verifier"  # decided at the gate...
    assert verdict.verifier_result is not None and not verdict.verifier_result.passed
    assert verdict.judge_scores is None  # ...and the jury never ran


@pytest.mark.parametrize("case_name,output_text,reference", MATH_BAD_CASES, ids=[c[0] for c in MATH_BAD_CASES])
def test_negative_control_tier3_math_dropped_before_jury(case_name, output_text, reference):
    router = VerificationRouter(spec=_spec(tier=3), verifier_hook=MATH_HOOK, judge=ExplodingJudge())
    cand = _cand(tier=3, input="What is 20 * 20?", output=output_text, reference=reference)
    verdict = router.verify(cand)
    assert verdict.kept is False
    assert verdict.stage == "verifier"
    assert verdict.judge_scores is None


def test_positive_control_tier3_reaches_jury_and_keeps_good_code():
    schema = load_record_schema("schemas/code_record.schema.json")
    hook = {**get_verifier_hook(schema), "timeout_s": CODE_HOOK_TIMEOUT}
    # min_score lowered: this control exercises the hybrid FLOW (gate -> jury -> keep),
    # not jury calibration of a terse code answer.
    spec = _spec(
        tier=3,
        modality="code",
        judge_config={"models": ["anthropic:claude-sonnet-4-6", "openai:gpt-judge"], "min_score": 1.0},
    )
    router = VerificationRouter(spec=spec, verifier_hook=hook, judge=HeuristicJudge())
    cand = _cand(tier=3, input="Double the input and explain.", output=GOOD_CODE, tests=CODE_TESTS)
    verdict = router.verify(cand)
    assert verdict.kept is True
    assert verdict.judge_scores is not None  # hybrid: gate passed AND jury graded
