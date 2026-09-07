"""Tiered verification router + quality filter + jury (Engine B 2.3), offline."""

from __future__ import annotations


from adh.contract import VerifiabilityTier, get_verifier_hook, load_record_schema
from adh.generation import Candidate
from adh.verification import (
    HeuristicJudge,
    QualityFilter,
    VerificationRouter,
    build_router,
)
from adh.verification.jury import PanelResult


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


def _cand(tier=2, *, input="Explain async macros clearly.", output="Async macros expand code at compile time to reduce boilerplate and keep call sites readable.", **kw):
    return Candidate(
        candidate_id="c1",
        job_id="job",
        aspect_id="a1",
        verifiability_tier=VerifiabilityTier(tier),
        input=input,
        output=output,
        **kw,
    )


# --------------------------------------------------------------------- quality
def test_quality_rejects_empty_and_short():
    q = QualityFilter()
    assert not q.check("").ok
    assert not q.check("hi").ok


def test_quality_rejects_symbol_soup_and_repetition():
    q = QualityFilter()
    assert not q.check("#### @@@@ ^^^^ %%%% $$$$ &&&&").ok
    assert not q.check(("na " * 200)).ok  # top-ngram repetition


def test_quality_accepts_prose():
    q = QualityFilter()
    assert q.check("This is a natural sentence that a person could plausibly have written.").ok


# --------------------------------------------------------------------- tier 1 (hard)
def test_tier1_passes_when_verifier_passes():
    schema = load_record_schema("schemas/code_record.schema.json")
    hook = {**get_verifier_hook(schema), "timeout_s": 30}
    cand = _cand(
        tier=1,
        input="Write echo",
        output="def echo(x):\n    return x\n",
        tests="def test_echo():\n    assert echo(3) == 3\n",
    )
    router = VerificationRouter(spec=_spec(tier=1, modality="code"), verifier_hook=hook)
    verdict = router.verify(cand)
    assert verdict.kept is True
    assert verdict.stage == "verifier"
    assert verdict.verifier_result and verdict.verifier_result.passed


def test_tier1_rejects_when_verifier_fails():
    schema = load_record_schema("schemas/code_record.schema.json")
    hook = {**get_verifier_hook(schema), "timeout_s": 30}
    cand = _cand(
        tier=1,
        input="Write echo",
        output="def echo(x):\n    return x + 1\n",
        tests="def test_echo():\n    assert echo(3) == 3\n",
    )
    router = VerificationRouter(spec=_spec(tier=1, modality="code"), verifier_hook=hook)
    assert router.verify(cand).kept is False


def test_tier1_without_hook_rejects():
    router = VerificationRouter(spec=_spec(tier=1, modality="code"), verifier_hook=None)
    assert router.verify(_cand(tier=1, output="def f(x):\n return x\n", tests="x")).kept is False


# --------------------------------------------------------------------- tier 2 (soft)
def test_tier2_jury_keeps_good_answer():
    router = build_router(_spec(tier=2), judge=HeuristicJudge())
    verdict = router.verify(_cand(tier=2))
    assert verdict.kept is True
    assert verdict.stage == "jury"
    assert verdict.judge_scores and verdict.judge_models == ("heuristic:judge",)


def test_tier2_jury_drops_low_scoring(monkeypatch):
    class HarshJudge:
        model_id = "heuristic:harsh"

        def evaluate(self, *, input, output, rubric, dimensions, reference=None):
            return PanelResult(scores={d: 1.0 for d in dimensions}, models=(self.model_id,))

    router = build_router(_spec(tier=2), judge=HarshJudge())
    verdict = router.verify(_cand(tier=2))
    assert verdict.kept is False
    assert verdict.judge_scores == {"helpfulness": 1.0, "correctness": 1.0, "clarity": 1.0}


def test_empty_input_rejected_before_jury():
    router = build_router(_spec(tier=2), judge=HeuristicJudge())
    assert router.verify(_cand(tier=2, input="   ")).kept is False


# --------------------------------------------------------------------- tier 3 (hybrid)
def test_tier3_gate_then_jury():
    schema = load_record_schema("schemas/code_record.schema.json")
    hook = {**get_verifier_hook(schema), "timeout_s": 30}
    good_code = "def solve(x):\n    return x * 2\n"
    cand = _cand(
        tier=3,
        input="Double the input and explain.",
        output=good_code,
        tests="def test_solve():\n    assert solve(2) == 4\n",
    )
    router = VerificationRouter(
        spec=_spec(tier=3, modality="code"), verifier_hook=hook, judge=HeuristicJudge()
    )
    verdict = router.verify(cand)
    # gate passed -> jury ran; verdict carries both the verifier result and judge scores
    assert verdict.verifier_result and verdict.verifier_result.passed
    assert verdict.judge_scores is not None


def test_tier3_failed_gate_skips_jury():
    schema = load_record_schema("schemas/code_record.schema.json")
    hook = {**get_verifier_hook(schema), "timeout_s": 30}
    cand = _cand(
        tier=3,
        input="Double the input.",
        output="def solve(x):\n    return x + 1\n",
        tests="def test_solve():\n    assert solve(2) == 4\n",
    )
    router = VerificationRouter(
        spec=_spec(tier=3, modality="code"), verifier_hook=hook, judge=HeuristicJudge()
    )
    verdict = router.verify(cand)
    assert verdict.kept is False
    assert verdict.stage == "verifier"  # rejected at the gate, jury never ran
    assert verdict.judge_scores is None


def test_tier3_without_hook_degrades_to_jury():
    """A hybrid spec whose schema carries no x-verifier (e.g. plain-text records like
    b6b176e6's) skips the gate and grades with the jury — the gate is a cost
    optimization for tier 3, not a requirement; rejecting everything stranded the run."""
    router = VerificationRouter(spec=_spec(tier=3), verifier_hook=None, judge=HeuristicJudge())
    verdict = router.verify(_cand(tier=3))
    assert verdict.kept is True
    assert verdict.stage == "jury"
    assert verdict.verifier_result is None


def test_tier3_without_hook_warns_once(caplog):
    """The degradation is loud but not per-candidate noisy."""
    import logging

    router = VerificationRouter(spec=_spec(tier=3), verifier_hook=None, judge=HeuristicJudge())
    with caplog.at_level(logging.WARNING, logger="adh.verification.router"):
        router.verify(_cand(tier=3))
        router.verify(_cand(tier=3))
    assert len([r for r in caplog.records if "x-verifier" in r.message]) == 1
