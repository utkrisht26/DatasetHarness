"""Diversity strategies in the SeedEngine (OUTLINE §2.1) — honoring ``diversity_strategy``.

Offline/hermetic: no network, no API keys. Strategy selection comes from a real
:class:`~adh.contract.JobSpec` (via the shared ``minimal_spec_dict`` fixture) or from
the engine constructor; every strategy is deterministic for a fixed seed.
"""

from __future__ import annotations

import logging
from datetime import datetime

from adh.contract import JobSpec, SourceChannel
from adh.corpus import CorpusChunk
from adh.discovery.models import Aspect
from adh.generation import (
    DEFAULT_PERSONAS,
    EVOLUTION_OPS,
    PersonaGridStrategy,
    PersonaHubStrategy,
    SeedEngine,
    persona_grid,
    resolve_strategy,
)

ASPECTS = (
    Aspect(aspect_id="a1", topic="async macros", rationale="", keywords=("async", "macro")),
    Aspect(aspect_id="a2", topic="error handling", rationale="", keywords=("result", "error")),
)


def _spec(minimal_spec_dict, *, diversity_strategy=None, seed=42) -> JobSpec:
    data = {
        **minimal_spec_dict,
        "description": "Rust async macro patterns and examples.",
        "motivation": "teach async macros",
        "generation_params": {"seed": seed},
    }
    if diversity_strategy is not None:
        data["diversity_strategy"] = diversity_strategy
    return JobSpec(**data)


def _chunk(aspect_id="a1", text="Async macros let you write concise code.") -> CorpusChunk:
    return CorpusChunk(
        chunk_id=f"job:{aspect_id}:abc",
        job_id="job",
        aspect_id=aspect_id,
        source_channel=SourceChannel.WEB,
        source_url="https://docs.rs/tokio",
        source_license="MIT",
        source_sha256="deadbeef",
        content_uri="corpus/job/a1.md",
        text_preview=text,
        n_tokens=8,
        fetched_at=datetime(2024, 1, 1),
    )


# --------------------------------------------------------------------- selection
def test_no_strategy_set_keeps_legacy_persona_grid(minimal_spec_dict):
    """Default ('persona+evol'/None) stays backward compatible: persona x difficulty."""
    for spec in (_spec(minimal_spec_dict), _spec(minimal_spec_dict, diversity_strategy="persona+evol")):
        seeds = SeedEngine(seed=3).plan(
            "job", aspects=ASPECTS, chunks=[_chunk("a1")], handoffs=[], n_seeds=6, spec=spec
        )
        assert len(seeds) == 6
        assert all(s.persona in DEFAULT_PERSONAS for s in seeds)
        assert all(s.evolution_op == "" and s.directive == "" for s in seeds)


def test_spec_selects_evol_instruct_and_persona_hub(minimal_spec_dict):
    evo_spec = _spec(minimal_spec_dict, diversity_strategy="evol_instruct")
    evo = SeedEngine(seed=3).plan(
        "job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=8, spec=evo_spec
    )
    assert {s.evolution_op for s in evo} <= set(EVOLUTION_OPS)

    hub_spec = _spec(minimal_spec_dict, diversity_strategy="persona_hub")
    hub = SeedEngine(seed=3).plan(
        "job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=8, spec=hub_spec
    )
    assert all(p in set(persona_grid()) for p in (s.persona for s in hub))


def test_constructor_strategy_wins_over_spec(minimal_spec_dict):
    spec = _spec(minimal_spec_dict, diversity_strategy="persona_hub")
    seeds = SeedEngine(seed=3, diversity_strategy="evol_instruct").plan(
        "job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=8, spec=spec
    )
    assert {s.evolution_op for s in seeds} == set(EVOLUTION_OPS)


def test_resolve_strategy_unknown_falls_back_to_default():
    strat = resolve_strategy("totally-bogus", personas=("p",), difficulties=("basic",))
    assert isinstance(strat, PersonaGridStrategy)
    known = resolve_strategy("persona_hub", personas=("p",), difficulties=("basic",))
    assert isinstance(known, PersonaHubStrategy)


# --------------------------------------------------------------------- determinism
def test_determinism_same_spec_and_seed(minimal_spec_dict):
    spec_a = _spec(minimal_spec_dict, diversity_strategy="persona_hub", seed=11)
    spec_b = _spec(minimal_spec_dict, diversity_strategy="persona_hub", seed=11)
    kwargs = dict(
        aspects=ASPECTS, chunks=[_chunk("a1"), _chunk("a2")], handoffs=[], n_seeds=20
    )
    a = SeedEngine().plan("job", spec=spec_a, **kwargs)
    b = SeedEngine().plan("job", spec=spec_b, **kwargs)
    assert a == b


def test_determinism_evol_instruct(minimal_spec_dict):
    spec = _spec(minimal_spec_dict, diversity_strategy="evol_instruct", seed=5)
    kwargs = dict(aspects=ASPECTS, chunks=[_chunk("a1")], handoffs=[], n_seeds=10)
    a = SeedEngine().plan("job", spec=spec, **kwargs)
    b = SeedEngine().plan("job", spec=spec, **kwargs)
    assert [s.seed_id for s in a] == [s.seed_id for s in b]
    assert a == b


# --------------------------------------------------------------------- persona hub
def test_persona_hub_grid_shape():
    grid = persona_grid()
    roles, expertise, intents = (
        PersonaHubStrategy.roles,
        PersonaHubStrategy.expertise,
        PersonaHubStrategy.intents,
    )
    assert len(grid) == len(roles) * len(expertise) * len(intents)
    assert len(set(grid)) == len(grid)  # full cross-product, no duplicates
    for role in roles:
        assert any(role in p for p in grid)
    for level in expertise:
        assert any(level in p for p in grid)
    for intent in intents:
        assert any(intent in p for p in grid)


def test_persona_hub_selection_varies_with_seed_but_is_reproducible(minimal_spec_dict):
    kwargs = dict(aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=6)
    low = SeedEngine().plan("job", spec=_spec(minimal_spec_dict, diversity_strategy="persona_hub", seed=101), **kwargs)
    high = SeedEngine().plan("job", spec=_spec(minimal_spec_dict, diversity_strategy="persona_hub", seed=202), **kwargs)
    again = SeedEngine().plan("job", spec=_spec(minimal_spec_dict, diversity_strategy="persona_hub", seed=101), **kwargs)
    assert [s.persona for s in again] == [s.persona for s in low]  # same seed -> same plan
    assert [s.persona for s in low] != [s.persona for s in high]  # different seed -> different mix


# --------------------------------------------------------------------- evol-instruct
def test_evol_instruct_produces_graded_variants(minimal_spec_dict):
    spec = _spec(minimal_spec_dict, diversity_strategy="evol_instruct")
    seeds = SeedEngine(seed=9).plan(
        "job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=24, spec=spec
    )
    assert seeds, "evolution must expand the unit"
    # every evolution op appears, and each maps onto a graded difficulty
    seen_ops = {s.evolution_op for s in seeds}
    assert seen_ops == set(EVOLUTION_OPS)
    by_op = {}
    for s in seeds:
        by_op.setdefault(s.evolution_op, set()).add(s.difficulty)
    assert by_op["concretize"] == {"basic"}
    assert by_op["constraints"] == {"intermediate"}
    assert {"advanced"} <= by_op["deepen"] and {"advanced"} <= by_op["counterexample"]
    # directives are populated and distinguishable per op
    directives = {s.directive for s in seeds}
    assert len(directives) >= len(EVOLUTION_OPS)
    assert all(d.strip() for d in directives)


def test_evolved_seeds_keep_grounding_signal(minimal_spec_dict):
    spec = _spec(minimal_spec_dict, diversity_strategy="evol_instruct")
    ch = _chunk("a2")
    seeds = SeedEngine(seed=9).plan(
        "job", aspects=ASPECTS[1:], chunks=[ch], handoffs=[], n_seeds=4, spec=spec
    )
    assert all(s.aspect_id == "a2" for s in seeds)
    assert all(s.grounding_uri == ch.content_uri for s in seeds)
    assert all(s.grounding_preview == ch.text_preview for s in seeds)
    assert all(s.derived_from == (ch.chunk_id,) for s in seeds)
    assert all(s.source_channel == SourceChannel.WEB for s in seeds)


# --------------------------------------------------------------------- fallback
def test_unknown_strategy_logs_warning_and_uses_default(minimal_spec_dict, caplog):
    spec = _spec(minimal_spec_dict, diversity_strategy="chaos_monkey")
    with caplog.at_level(logging.WARNING, logger="adh.generation.seed"):
        seeds = SeedEngine(seed=3).plan(
            "job", aspects=ASPECTS[:1], chunks=[_chunk("a1")], handoffs=[], n_seeds=6, spec=spec
        )
    assert any("chaos_monkey" in rec.message and rec.levelno == logging.WARNING for rec in caplog.records)
    # fell back to the legacy grid: plain personas, no evolution metadata
    assert all(s.persona in DEFAULT_PERSONAS for s in seeds)
    assert all(s.evolution_op == "" for s in seeds)


# --------------------------------------------------------------------- legacy seams
def test_plan_without_spec_still_works_like_before():
    seeds = SeedEngine(seed=7).plan("job", aspects=ASPECTS, chunks=[_chunk("a1")], handoffs=[], n_seeds=4)
    assert len(seeds) == 4
    assert all(s.persona in DEFAULT_PERSONAS for s in seeds)
