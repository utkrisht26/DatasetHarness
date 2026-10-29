"""Cost & model routing (OUTLINE §3.2) — cheap bulk generation, frontier escalation.

The generator ladder from ``spec.generator_models`` is applied at data scale: easy
seeds generate on the cheapest tier, ``advanced`` seeds start at the frontier tier
(hard cases escalate up front), and any tier that fails or comes back empty hands
the seed to the next tier. Fully offline via injected fake generators.
"""

from __future__ import annotations

from adh.contract import JobSpec, SourceChannel, VerifiabilityTier
from adh.generation.models import Candidate, SeedPoint, make_candidate_id
from adh.generation.routing import RoutedGenerator


# ----------------------------------------------------------------- fakes
class FakeGen:
    """Records the seeds it was asked about; scripted per-seed results."""

    def __init__(self, name: str, results: dict[str, list] | None = None, *, fail=False):
        self.name = name
        self.results = results or {}
        self.fail = fail
        self.asked: list[SeedPoint] = []

    def generate(self, seed, *, spec, store=None):
        self.asked.append(seed)
        if self.fail:
            raise RuntimeError(f"{self.name} exploded")
        return self.results.get(seed.seed_id, [])


def _seed(difficulty: str = "basic", sid: str = "s1") -> SeedPoint:
    return SeedPoint(
        seed_id=sid,
        job_id="demo",
        aspect_id="a0",
        topic="rust async",
        keywords=("rust",),
        persona="expert",
        difficulty=difficulty,
        source_channel=SourceChannel.SYNTHESIS,
    )


def _cand(seed: SeedPoint) -> Candidate:
    return Candidate(
        candidate_id=make_candidate_id(seed.job_id, seed.aspect_id, "in", "out"),
        job_id=seed.job_id,
        aspect_id=seed.aspect_id,
        verifiability_tier=VerifiabilityTier.SOFT,
        input="in",
        output="out",
    )


def _spec(minimal_spec_dict) -> JobSpec:
    d = {**minimal_spec_dict, "description": "Rust async macros.", "motivation": "teach"}
    return JobSpec(**d)


# ----------------------------------------------------------------- routing policy
def test_basic_seed_uses_cheapest_tier_only(minimal_spec_dict):
    cheap = FakeGen("cheap", {"s1": [_cand(_seed())]})
    frontier = FakeGen("frontier")
    routed = RoutedGenerator(tiers=(cheap, frontier))
    spec = _spec(minimal_spec_dict)

    out = routed.generate(_seed("basic"), spec=spec)

    assert len(out) == 1 and not frontier.asked
    assert routed.calls == {"tier0": 1}


def test_advanced_seed_starts_at_the_frontier_tier(minimal_spec_dict):
    cheap = FakeGen("cheap")
    frontier = FakeGen("frontier", {"s2": [_cand(_seed("advanced"))]})
    routed = RoutedGenerator(tiers=(cheap, frontier))
    spec = _spec(minimal_spec_dict)

    out = routed.generate(_seed("advanced", "s2"), spec=spec)

    assert len(out) == 1
    assert not cheap.asked  # hard cases skip straight to the frontier model
    assert routed.calls == {"tier1": 1}


def test_empty_result_escalates_to_next_tier(minimal_spec_dict):
    cheap = FakeGen("cheap", {"s3": []})  # unparseable/refusal -> no candidates
    mid = FakeGen("mid", {"s3": [_cand(_seed(sid="s3"))]})
    routed = RoutedGenerator(tiers=(cheap, mid))
    spec = _spec(minimal_spec_dict)

    out = routed.generate(_seed(sid="s3"), spec=spec)

    assert len(out) == 1
    assert routed.calls == {"tier0": 1, "tier1": 1}
    assert routed.escalations == 1


def test_raising_tier_escalates_instead_of_killing_the_seed(minimal_spec_dict):
    boom = FakeGen("boom", fail=True)
    solid = FakeGen("solid", {"s4": [_cand(_seed(sid="s4"))]})
    routed = RoutedGenerator(tiers=(boom, solid))
    spec = _spec(minimal_spec_dict)

    out = routed.generate(_seed(sid="s4"), spec=spec)

    assert len(out) == 1


def test_all_tiers_exhausted_returns_empty_and_counts_it(minimal_spec_dict):
    a, b = FakeGen("a"), FakeGen("b")
    routed = RoutedGenerator(tiers=(a, b))
    spec = _spec(minimal_spec_dict)

    assert routed.generate(_seed(sid="s5"), spec=spec) == []
    assert routed.exhausted == 1


def test_a_skip_is_final_not_an_exhausted_or_escalated_seed(minimal_spec_dict):
    # Live run rust-async-50 reported "generator exhausted 71/150 seed(s) — check
    # provider health" when every one of those was a deliberate junk-grounding skip.
    import pytest

    from adh.generation import SeedSkipped

    class Skipper(FakeGen):
        def generate(self, seed, *, spec, store=None):
            self.asked.append(seed)
            raise SeedSkipped("navigation page")

    cheap, frontier = Skipper("cheap"), FakeGen("frontier")
    routed = RoutedGenerator(tiers=(cheap, frontier))

    with pytest.raises(SeedSkipped, match="navigation page"):
        routed.generate(_seed(sid="s7"), spec=_spec(minimal_spec_dict))

    assert len(cheap.asked) == 1  # no retries: a skip is a verdict, not a fault
    assert frontier.asked == []  # the grounding is junk for every tier
    assert routed.exhausted == 0 and routed.escalations == 0


def test_single_tier_ladder_degenerates_to_plain_dispatch(minimal_spec_dict):
    only = FakeGen("only", {"s6": [_cand(_seed(sid="s6"))]})
    routed = RoutedGenerator(tiers=(only,))
    spec = _spec(minimal_spec_dict)

    assert routed.generate(_seed(sid="s6"), spec=spec)
    assert routed.escalations == 0


# ----------------------------------------------------------------- production wiring
def test_routed_from_models_builds_lazy_llm_tiers():
    """The production helper defers chat-model construction until a tier is used."""
    from adh.generation.routing import routed_from_models

    tiers = routed_from_models(("openai:gpt-4o-mini", "anthropic:claude-opus-4-5"))
    assert callable(tiers[0]) and callable(tiers[1])


def test_build_generator_wires_the_full_ladder(minimal_spec_dict, monkeypatch):
    # fake credentials: the probe only checks the provider integration constructs
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.setenv(var, "test-placeholder-not-a-real-key")
    from adh.generation.generator import build_generator

    d = {
        **minimal_spec_dict,
        "description": "Rust async macros.",
        "motivation": "teach",
        "generator_models": ["openai:gpt-4o-mini", "anthropic:claude-opus-4-5"],
    }
    gen = build_generator(JobSpec(**d))
    from adh.generation.routing import RoutedGenerator as RG

    assert isinstance(gen, RG)


def test_build_generator_no_models_keeps_heuristic(minimal_spec_dict):
    from adh.generation.generator import HeuristicGenerator, build_generator

    d = {**minimal_spec_dict, "description": "Rust async macros.", "motivation": "teach"}
    d["generator_models"] = []
    assert isinstance(build_generator(JobSpec(**d)), HeuristicGenerator)
