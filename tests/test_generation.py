"""Seed engine + record generator (Engine B 2.1/2.2), exercised offline."""

from __future__ import annotations

from datetime import datetime, timezone


from adh.contract import SourceChannel, VerifiabilityTier
from adh.corpus import CorpusChunk, SynthesisHandoff
from adh.discovery.models import Aspect
from adh.generation import (
    HeuristicGenerator,
    NoveltyGate,
    SeedEngine,
    build_generator,
    coverage_report,
)
from adh.verifiers.code import run_pytest
from adh.verifiers.math import check_equal


def _spec(tier=2, modality="text", **kw):
    """A valid JobSpec honoring the Tier 2/3 judge_config + decontam invariants."""
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


def _chunk(aspect_id="a1", text="Async macros let you write concise code.", channel=SourceChannel.WEB):
    return CorpusChunk(
        chunk_id=f"job:{aspect_id}:abc",
        job_id="job",
        aspect_id=aspect_id,
        source_channel=channel,
        source_url="https://docs.rs/tokio",
        source_license="MIT",
        source_sha256="deadbeef",
        content_uri="",
        text_preview=text,
        n_tokens=8,
        fetched_at=datetime.now(timezone.utc),
    )


ASPECTS = (
    Aspect(aspect_id="a1", topic="async macros", rationale="", keywords=("async", "macro")),
    Aspect(aspect_id="a2", topic="error handling", rationale="", keywords=("result", "error")),
)


# --------------------------------------------------------------------- seed engine
def test_seed_engine_balances_across_aspects():
    chunks = [_chunk("a1"), _chunk("a2")]
    seeds = SeedEngine().plan("job", aspects=ASPECTS, chunks=chunks, handoffs=[], n_seeds=6)
    assert len(seeds) == 6
    by_aspect = {}
    for s in seeds:
        by_aspect.setdefault(s.aspect_id, 0)
        by_aspect[s.aspect_id] += 1
    # round-robin => the two aspects are within one of each other
    assert abs(by_aspect["a1"] - by_aspect["a2"]) <= 1


def test_seed_engine_is_deterministic():
    a = SeedEngine(seed=7).plan("job", aspects=ASPECTS, chunks=[_chunk("a1")], handoffs=[], n_seeds=4)
    b = SeedEngine(seed=7).plan("job", aspects=ASPECTS, chunks=[_chunk("a1")], handoffs=[], n_seeds=4)
    assert [s.seed_id for s in a] == [s.seed_id for s in b]


def test_seed_engine_covers_synthesis_handoffs():
    handoff = SynthesisHandoff(job_id="job", aspect_id="a3", topic="traits", keywords=("trait",))
    seeds = SeedEngine().plan("job", aspects=(), chunks=[], handoffs=[handoff], n_seeds=3)
    assert seeds and all(s.aspect_id == "a3" for s in seeds)
    assert all(s.source_channel == SourceChannel.SYNTHESIS for s in seeds)


def test_seed_engine_zero_seeds():
    assert SeedEngine().plan("job", aspects=ASPECTS, chunks=[_chunk()], handoffs=[], n_seeds=0) == []


# --------------------------------------------------------------------- novelty gate
def test_novelty_gate_rejects_near_duplicate():
    gate = NoveltyGate(threshold=0.6, ngram=2)
    assert gate.is_novel("explain async macros at a basic level")
    assert not gate.is_novel("explain async macros at a basic level")  # identical
    assert gate.is_novel("describe error handling with the Result type")  # different


# --------------------------------------------------------------------- heuristic generator
def test_heuristic_generator_soft_grounds_in_corpus():
    seeds = SeedEngine().plan("job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=1)
    cands = HeuristicGenerator().generate(seeds[0], spec=_spec(tier=2))
    assert len(cands) == 1
    c = cands[0]
    assert c.input and c.output
    assert c.verifiability_tier == VerifiabilityTier.SOFT
    assert c.source_channel == SourceChannel.WEB
    assert c.derived_from == ("job:a1:abc",)


def test_heuristic_generator_hard_code_is_verifier_passing():
    spec = _spec(tier=1, modality="code", record_schema_ref="schemas/code_record.schema.json")
    seeds = SeedEngine().plan("job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=1)
    c = HeuristicGenerator().generate(seeds[0], spec=spec)[0]
    assert c.tests is not None
    # the offline generator must emit a genuinely passing Tier-1 pair
    assert run_pytest(c.to_verifier_request(), {"timeout_s": 30}).passed is True


def test_heuristic_generator_hard_math_is_verifier_passing():
    spec = _spec(tier=1, modality="text", task_categories=["math"])
    handoff = SynthesisHandoff(job_id="job", aspect_id="m1", topic="arithmetic", keywords=("math",))
    seeds = SeedEngine().plan("job", aspects=(), chunks=[], handoffs=[handoff], n_seeds=1)
    c = HeuristicGenerator().generate(seeds[0], spec=spec)[0]
    assert c.reference is not None
    assert check_equal(c.to_verifier_request(), {}).passed is True


def test_build_generator_falls_back_without_models():
    assert isinstance(build_generator(_spec()), HeuristicGenerator)


# --------------------------------------------------------------------- coverage
def test_coverage_report_flags_thin_aspects():
    report = coverage_report(["a1", "a1", "a1"], ASPECTS, target_records=8)
    assert report.per_aspect["a1"] == 3
    assert report.per_aspect["a2"] == 0
    assert "a2" in report.thin_aspects
    assert "a1" not in report.thin_aspects
