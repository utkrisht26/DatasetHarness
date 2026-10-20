"""Seed engine + record generator (Engine B 2.1/2.2), exercised offline."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest


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


def _chunk(
    aspect_id="a1", text="Async macros let you write concise code.", channel=SourceChannel.WEB
):
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
    a = SeedEngine(seed=7).plan(
        "job", aspects=ASPECTS, chunks=[_chunk("a1")], handoffs=[], n_seeds=4
    )
    b = SeedEngine(seed=7).plan(
        "job", aspects=ASPECTS, chunks=[_chunk("a1")], handoffs=[], n_seeds=4
    )
    assert [s.seed_id for s in a] == [s.seed_id for s in b]


def test_seed_engine_covers_synthesis_handoffs():
    handoff = SynthesisHandoff(job_id="job", aspect_id="a3", topic="traits", keywords=("trait",))
    seeds = SeedEngine().plan("job", aspects=(), chunks=[], handoffs=[handoff], n_seeds=3)
    assert seeds and all(s.aspect_id == "a3" for s in seeds)
    assert all(s.source_channel == SourceChannel.SYNTHESIS for s in seeds)


def test_seed_engine_zero_seeds():
    assert (
        SeedEngine().plan("job", aspects=ASPECTS, chunks=[_chunk()], handoffs=[], n_seeds=0) == []
    )


# --------------------------------------------------------------------- novelty gate
def test_novelty_gate_rejects_near_duplicate():
    gate = NoveltyGate(threshold=0.6, ngram=2)
    assert gate.is_novel("explain async macros at a basic level")
    assert not gate.is_novel("explain async macros at a basic level")  # identical
    assert gate.is_novel("describe error handling with the Result type")  # different


# --------------------------------------------------------------------- heuristic generator
def test_heuristic_generator_soft_grounds_in_corpus():
    seeds = SeedEngine().plan(
        "job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=1
    )
    cands = HeuristicGenerator().generate(seeds[0], spec=_spec(tier=2))
    assert len(cands) == 1
    c = cands[0]
    assert c.input and c.output
    assert c.verifiability_tier == VerifiabilityTier.SOFT
    assert c.source_channel == SourceChannel.WEB
    assert c.derived_from == ("job:a1:abc",)


def test_heuristic_generator_hard_code_is_verifier_passing():
    spec = _spec(tier=1, modality="code", record_schema_ref="schemas/code_record.schema.json")
    seeds = SeedEngine().plan(
        "job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=1
    )
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


# ------------------------------------------------------- env-configured model ladder
def _placeholder_llm_keys(monkeypatch):
    """Fake credentials so provider integrations construct (no network until invoke)."""
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "test-placeholder-not-a-real-key")


def test_build_generator_env_ladder_overrides_the_spec(monkeypatch):
    """ADH_GENERATOR_MODELS is the deployment-level redirect, Engine-A parity."""
    import adh.llm as llm
    from adh.generation.routing import RoutedGenerator

    monkeypatch.setenv(llm.models.GENERATOR_MODELS_ENV, "openai:gen-a, anthropic:gen-b")
    _placeholder_llm_keys(monkeypatch)

    gen = build_generator(_spec(generator_models=["openai:gpt-4o-mini"]))

    assert isinstance(gen, RoutedGenerator)
    assert gen.tiers[0]().model_id == "openai:gen-a"
    assert gen.tiers[1]().model_id == "anthropic:gen-b"


def test_build_generator_without_env_keeps_the_spec_ladder(monkeypatch):
    import adh.llm as llm
    from adh.generation.routing import RoutedGenerator

    monkeypatch.delenv(llm.models.GENERATOR_MODELS_ENV, raising=False)
    _placeholder_llm_keys(monkeypatch)

    gen = build_generator(_spec(generator_models=["openai:gpt-4o-mini"]))

    assert isinstance(gen, RoutedGenerator)
    assert gen.tiers[0]().model_id == "openai:gpt-4o-mini"


def test_build_generator_env_override_beats_empty_spec_too(monkeypatch):
    """A deployment ladder applies even where a spec configures none (heuristic today)."""
    import adh.llm as llm
    from adh.generation.routing import RoutedGenerator

    monkeypatch.setenv(llm.models.GENERATOR_MODELS_ENV, "openai:gen-a")
    _placeholder_llm_keys(monkeypatch)

    assert isinstance(build_generator(_spec()), RoutedGenerator)


# ------------------------------------------------------- tier retry resilience
def _one_seed():
    return SeedEngine().plan(
        "job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=1
    )[0]


def test_routed_generator_retries_transient_tier_failures(monkeypatch):
    """A provider blip retries within its tier instead of zeroing the build — the
    b6b176e6 lesson: one bad upstream minute exhausted a single-tier ladder silently."""
    import adh.generation.routing as routing

    monkeypatch.setattr(routing, "_sleep", lambda *_: None)

    class Flaky:
        def __init__(self):
            self.calls = 0

        def generate(self, seed, *, spec, store=None):
            self.calls += 1
            if self.calls < 3:
                raise RuntimeError("upstream 502")
            return ["candidate"]

    flaky = Flaky()
    gen = routing.RoutedGenerator(tiers=(lambda: flaky,))
    seed = _one_seed()

    assert gen.generate(seed, spec=_spec()) == ["candidate"]
    assert gen.exhausted == 0
    assert flaky.calls == 3  # two faults absorbed + the successful attempt
    assert gen.calls == {"tier0": 1}  # calls counts seeds dispatched, not attempts


def test_routed_generator_escalates_after_retry_exhaustion(monkeypatch):
    """Persistent tier failure still escalates upward — retries never mask a dead tier."""
    import adh.generation.routing as routing

    monkeypatch.setattr(routing, "_sleep", lambda *_: None)

    class Boom:
        def generate(self, seed, *, spec, store=None):
            raise RuntimeError("down")

    class Ok:
        def generate(self, seed, *, spec, store=None):
            return ["from-tier1"]

    gen = routing.RoutedGenerator(tiers=(Boom, Ok))

    assert gen.generate(_one_seed(), spec=_spec()) == ["from-tier1"]
    assert gen.escalations == 1
    assert gen.calls == {"tier0": 1, "tier1": 1}


def test_routed_generator_does_not_replay_empty_tier_results():
    """An empty-but-clean result is a quality verdict, not a fault — escalate at once."""
    import adh.generation.routing as routing

    class Empty:
        def __init__(self):
            self.calls = 0

        def generate(self, seed, *, spec, store=None):
            self.calls += 1
            return []

    empty = Empty()
    gen = routing.RoutedGenerator(tiers=(lambda: empty,))

    assert gen.generate(_one_seed(), spec=_spec()) == []
    assert empty.calls == 1
    assert gen.exhausted == 1


# ------------------------------------------------------- unparseable-output re-ask
class _ScriptedModel:
    """Returns queued replies, recording each invoke's message list."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages, **params):
        self.calls.append(list(messages))
        return type("R", (), {"content": self.replies.pop(0)})()


def _scripted_llm(replies):
    from adh.generation.generator import LlmGenerator

    gen = LlmGenerator.__new__(LlmGenerator)  # bypass __post_init__ (no provider build)
    gen.model_id = "openai:test"
    gen.usage = None
    gen.max_grounding_chars = 4000
    gen._model = _ScriptedModel(replies)
    return gen


def test_llm_generator_reasks_once_after_prose_reply():
    """A prose reply gets exactly one corrective re-ask instead of wasting the call."""
    import json

    good = json.dumps({"input": "What is i2v?", "output": "Image-to-video generation."})
    gen = _scripted_llm(["Sure! This record is about dancing models.", good])

    cands = gen.generate(_one_seed(), spec=_spec(tier=2))

    assert len(cands) == 1
    assert cands[0].input == "What is i2v?"
    assert len(gen._model.calls) == 2
    # the corrective turn replays the prior reply and demands JSON only
    assert len(gen._model.calls[1]) == 4
    assert "not valid JSON" in gen._model.calls[1][-1].content


def test_llm_generator_gives_up_after_one_reask(caplog):
    import logging

    gen = _scripted_llm(["still no json here", "nope"])

    with caplog.at_level(logging.WARNING, logger="adh.generation.generator"):
        assert gen.generate(_one_seed(), spec=_spec(tier=2)) == []
    assert len(gen._model.calls) == 2
    # the surrender names the seed AND shows what the model actually said
    heads = [r for r in caplog.records if "unparseable" in r.message]
    assert heads and "still no json here" in heads[-1].getMessage()


@pytest.mark.parametrize(
    "truncated",
    [
        "",  # all budget spent reasoning (live, max_tokens=1024 and 8192)
        '{"input":"Explain Rust\'s async execution model for an expert',  # cut mid-JSON
    ],
)
def test_llm_generator_retries_reasoning_truncation_with_more_headroom(caplog, truncated):
    """Live runs: a reasoning model hit max_tokens (finish_reason 'length') with no
    content, or mid-way through the JSON; the JSON-correction re-ask at the same
    budget failed identically. Truncation must instead retry with a larger budget."""
    import json
    import logging

    class _Reply:
        def __init__(self, content, finish_reason):
            self.content = content
            self.response_metadata = {"finish_reason": finish_reason}

    class _BudgetModel:
        def __init__(self):
            self.calls = []
            self.replies = [
                _Reply(truncated, "length"),
                _Reply(
                    json.dumps({"input": "What does Pin guarantee?", "output": "No moves."}), "stop"
                ),
            ]

        def invoke(self, messages, **params):
            self.calls.append((list(messages), params))
            return self.replies.pop(0)

    gen = _scripted_llm([])
    gen._model = _BudgetModel()
    spec = _spec(tier=2, generation_params={"max_tokens": 16384})

    with caplog.at_level(logging.WARNING, logger="adh.generation.generator"):
        cands = gen.generate(_one_seed(), spec=spec)

    assert len(cands) == 1
    (first_msgs, first), (retry_msgs, retry) = gen._model.calls
    assert retry["max_tokens"] > first.get("max_tokens", 0)
    assert retry_msgs == first_msgs  # same prompt, not the JSON-correction turn
    assert any("max_tokens" in r.getMessage() for r in caplog.records)


def test_llm_generator_honors_skip_for_off_topic_context(caplog):
    """A {"skip": ...} reply is a clean opt-out: no candidate, no corrective re-ask."""
    import json
    import logging

    gen = _scripted_llm([json.dumps({"skip": "context is a job listing, not Rust async"})])

    with caplog.at_level(logging.INFO, logger="adh.generation.generator"):
        assert gen.generate(_one_seed(), spec=_spec(tier=2)) == []
    assert len(gen._model.calls) == 1
    assert any("job listing" in r.getMessage() for r in caplog.records)


def test_grounded_prompt_demands_standalone_records_and_offers_skip():
    from adh.generation.prompts import generation_prompt

    prompt = generation_prompt(_one_seed(), _spec(tier=2), grounding="Tokio steals tasks.")
    assert "standalone" in prompt
    assert '"skip"' in prompt
    # the old wording made the context the *subject* of the record
    assert "Ground the record ONLY in this source context" not in prompt


def test_llm_generator_clean_json_needs_no_reask():
    import json

    gen = _scripted_llm([json.dumps({"input": "Q?", "output": "A."})])

    cands = gen.generate(_one_seed(), spec=_spec(tier=2))

    assert len(cands) == 1
    assert len(gen._model.calls) == 1


# --------------------------------------------------------------------- coverage
def test_coverage_report_flags_thin_aspects():
    report = coverage_report(["a1", "a1", "a1"], ASPECTS, target_records=8)
    assert report.per_aspect["a1"] == 3
    assert report.per_aspect["a2"] == 0
    assert "a2" in report.thin_aspects
    assert "a1" not in report.thin_aspects


# ------------------------------------------------------- candidate media threading
def _media_ref():
    from adh.contract import MediaRef

    return MediaRef(
        path="media/" + "a" * 64 + ".mp4",
        sha256="a" * 64,
        bytes=2048,
        duration_s=6.0,
        width=1080,
        height=1920,
        format="mp4",
        attribution="Pexels: A Videographer",
    )


def _video_candidate(**over):
    from adh.generation.models import make_candidate_id

    base = dict(
        candidate_id=make_candidate_id("job", "a1", "scene", "caption"),
        job_id="job",
        aspect_id="a1",
        verifiability_tier=VerifiabilityTier.HYBRID,
        input="a couple dancing outdoors",
        output="A couple dances outdoors at golden hour.",
        media=_media_ref(),
        staged_path="/runs/j/media_staging/job/" + "a" * 64 + ".mp4",
        model_id="packager:adh",
        pipeline_step="media-packaging",
    )
    base.update(over)
    from adh.generation.models import Candidate

    return Candidate(**base)


def test_candidate_threads_media_into_verifier_request_extras():
    req = _video_candidate().to_verifier_request()

    assert req.extras["media_abs_path"] == _video_candidate().staged_path
    assert req.extras["media"]["sha256"] == "a" * 64
    assert req.extras["media"]["duration_s"] == 6.0


def test_text_candidate_keeps_empty_verifier_extras():
    seeds = SeedEngine().plan(
        "job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=1
    )
    cand = HeuristicGenerator().generate(seeds[0], spec=_spec())[0]

    assert cand.to_verifier_request().extras == {}


def test_candidate_media_round_trips_through_dicts():
    cand = _video_candidate()

    restored = type(cand).from_dict(cand.to_dict())

    assert restored.media == cand.media
    assert restored.staged_path == cand.staged_path


def test_legacy_candidate_dicts_without_media_still_parse():
    from adh.generation.models import Candidate

    legacy = _video_candidate().to_dict()
    del legacy["media"]
    del legacy["staged_path"]

    restored = Candidate.from_dict(legacy)

    assert restored.media is None
    assert restored.staged_path is None


def test_base_record_fields_carry_media():
    assert _video_candidate().base_record_fields()["media"] == _media_ref()
    text_candidate = HeuristicGenerator().generate(_one_seed(), spec=_spec())[0]
    assert text_candidate.base_record_fields()["media"] is None
