"""Seed / taxonomy engine (OUTLINE §2.1) — diversity is the systemic risk.

Turns the discovered aspects + acquired corpus into a *structured sampling space*
rather than chasing raw volume: every seed crosses an aspect's grounding with a
persona and a difficulty, distributed round-robin across aspects so no single topic
dominates. A :class:`NoveltyGate` (shingle-Jaccard, a deterministic ROUGE-L proxy)
implements the Self-Instruct novelty filter, and :func:`coverage_report` tracks how
the accepted set covers the taxonomy so the driver can ask Engine A to re-discover
thin aspects (the coverage-gap loop).

Diversity strategies (JobSpec ``diversity_strategy``): how each grounding unit
expands into :class:`SeedPoint` s is a pluggable port (:class:`DiversityStrategy`):

* ``persona+evol`` (the contract default) / ``None`` — the historical
  persona x difficulty cross-product (:class:`PersonaGridStrategy`);
* ``evol_instruct`` — evolve the unit's instruction along graded axes: concretize,
  add constraints, deepen, counterexample (:class:`EvolInstructStrategy`);
* ``persona_hub`` — synthesize composite personas from a deterministic
  role x expertise x intent grid, selection seeded by the job seed
  (:class:`PersonaHubStrategy`).

Unknown strategy values log a warning and fall back to the default. Everything is
deterministic: the only randomness is :class:`random.Random` keyed by the seed.
"""

from __future__ import annotations

import logging
import random
import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from adh.contract import JobSpec, SourceChannel
from adh.corpus import CorpusChunk, SynthesisHandoff
from adh.discovery.models import Aspect

from .models import DEFAULT_DIFFICULTIES, DEFAULT_PERSONAS, SeedPoint

logger = logging.getLogger(__name__)

_WORD_RE = re.compile(r"[a-z0-9]+")

# ------------------------------------------------------------------ strategy names
DEFAULT_DIVERSITY_STRATEGY = "persona+evol"  # the JobSpec contract default
EVOL_INSTRUCT_STRATEGY = "evol_instruct"
PERSONA_HUB_STRATEGY = "persona_hub"

# Evol-Instruct operators, ordered from concretization to counterfactual reasoning —
# the order IS the grading (basic -> advanced) on the difficulty axis.
EVOLUTION_OPS = ("concretize", "constraints", "deepen", "counterexample")

_OP_DIFFICULTY = {"concretize": 0, "constraints": 1, "deepen": 2, "counterexample": 2}

_EVOL_DIRECTIVES: dict[str, str] = {
    "concretize": (
        "Concretize the request: anchor it in one specific scenario or worked example "
        "involving {kw0}, with concrete inputs and expected behavior."
    ),
    "constraints": (
        "Add explicit constraints the answer must satisfy (stay within {kw0}, justify "
        "step by step, no hand-waving)."
    ),
    "deepen": (
        "Deepen the problem beyond the basics: require multi-step reasoning about "
        "{topic} that connects at least two of its concepts."
    ),
    "counterexample": (
        "Ask for a counterexample: a case where the naive approach to {topic} breaks, "
        "and why the failure happens."
    ),
}

# Persona-Hub grid axes: composite personas = role x expertise x intent.
HUB_ROLES = ("a backend engineer", "a data scientist", "an educator", "a technical writer")
HUB_EXPERTISE = ("novice", "practitioner", "expert")
HUB_INTENTS = ("learning", "debugging", "evaluating", "teaching")


def persona_grid(
    roles: tuple[str, ...] = HUB_ROLES,
    expertise: tuple[str, ...] = HUB_EXPERTISE,
    intents: tuple[str, ...] = HUB_INTENTS,
) -> tuple[str, ...]:
    """The deterministic persona-hub grid: role x expertise x intent cross-product."""
    return tuple(f"{role}, {level}-level, motivated by {intent}"
                 for role in roles for level in expertise for intent in intents)


# ------------------------------------------------------------------ strategy port
@dataclass(frozen=True)
class UnitContext:
    """One grounding unit (corpus chunk, synthesis handoff, or bare aspect).

    Carries everything a strategy needs to expand the unit into SeedPoints,
    including a per-unit :class:`random.Random` (keyed by job seed + aspect + unit
    ordinal) so strategies that sample stay deterministic per run.
    """

    unit_prefix: str  # stable id prefix for seed_ids emitted from this unit
    job_id: str
    aspect_id: str
    topic: str
    keywords: tuple[str, ...]
    source_channel: SourceChannel
    grounding_uri: str | None = None
    grounding_preview: str = ""
    source_url: str | None = None
    source_license: str = "unknown"
    source_sha256: str | None = None
    derived_from: tuple[str, ...] = ()
    rng: random.Random = field(default_factory=random.Random)


@runtime_checkable
class DiversityStrategy(Protocol):
    """Port: expand one grounding unit into SeedPoints (the sampling-space atoms)."""

    def expand(self, ctx: UnitContext) -> list[SeedPoint]:
        """Produce zero or more seed points for one grounding unit."""
        ...


@dataclass(frozen=True)
class PersonaGridStrategy:
    """Default expansion: the persona x difficulty cross-product (OUTLINE §2.1)."""

    personas: tuple[str, ...] = DEFAULT_PERSONAS
    difficulties: tuple[str, ...] = DEFAULT_DIFFICULTIES

    def combos(self):
        for p in self.personas:
            for d in self.difficulties:
                yield p, d

    def expand(self, ctx: UnitContext) -> list[SeedPoint]:
        return [
            SeedPoint(
                seed_id=f"{ctx.unit_prefix}:{i}",
                job_id=ctx.job_id,
                aspect_id=ctx.aspect_id,
                topic=ctx.topic,
                keywords=ctx.keywords,
                persona=persona,
                difficulty=difficulty,
                source_channel=ctx.source_channel,
                grounding_uri=ctx.grounding_uri,
                grounding_preview=ctx.grounding_preview,
                source_url=ctx.source_url,
                source_license=ctx.source_license,
                source_sha256=ctx.source_sha256,
                derived_from=ctx.derived_from,
            )
            for i, (persona, difficulty) in enumerate(self.combos())
        ]


@dataclass(frozen=True)
class EvolInstructStrategy:
    """Evol-Instruct expansion: evolve the unit's instruction along graded axes.

    Each grounding unit is evolved once per (operator, persona) pair. Operators run
    concretize -> constraints -> deepen -> counterexample, each mapped onto the
    difficulty axis (basic/intermediate/advanced), so a unit yields *graded*
    variants rather than uniform noise. Every seed carries the machine-readable
    ``evolution_op`` plus a human ``directive`` a generator can fold into its prompt.
    """

    personas: tuple[str, ...] = DEFAULT_PERSONAS
    difficulties: tuple[str, ...] = DEFAULT_DIFFICULTIES
    ops: tuple[str, ...] = EVOLUTION_OPS

    def expand(self, ctx: UnitContext) -> list[SeedPoint]:
        seeds: list[SeedPoint] = []
        j = 0
        for op in self.ops:
            grade = min(_OP_DIFFICULTY.get(op, 0), len(self.difficulties) - 1)
            difficulty = self.difficulties[grade]
            directive = self._directive(op, ctx)
            for persona in self.personas:
                seeds.append(
                    SeedPoint(
                        seed_id=f"{ctx.unit_prefix}:evo:{j}",
                        job_id=ctx.job_id,
                        aspect_id=ctx.aspect_id,
                        topic=ctx.topic,
                        keywords=ctx.keywords,
                        persona=persona,
                        difficulty=difficulty,
                        source_channel=ctx.source_channel,
                        grounding_uri=ctx.grounding_uri,
                        grounding_preview=ctx.grounding_preview,
                        source_url=ctx.source_url,
                        source_license=ctx.source_license,
                        source_sha256=ctx.source_sha256,
                        derived_from=ctx.derived_from,
                        evolution_op=op,
                        directive=directive,
                    )
                )
                j += 1
        return seeds

    @staticmethod
    def _directive(op: str, ctx: UnitContext) -> str:
        kw0 = ctx.keywords[0] if ctx.keywords else ctx.topic
        return _EVOL_DIRECTIVES.get(op, "{topic}").format(topic=ctx.topic, kw0=kw0)


@dataclass(frozen=True)
class PersonaHubStrategy:
    """Persona-Hub expansion: composite personas from a deterministic grid.

    Personas are synthesized as ``role x expertise x intent`` composites
    (:func:`persona_grid`); the job seed drives a per-unit offset into the grid so
    different seeds sample different regions while every run stays reproducible.
    """

    roles: tuple[str, ...] = HUB_ROLES
    expertise: tuple[str, ...] = HUB_EXPERTISE
    intents: tuple[str, ...] = HUB_INTENTS
    personas: tuple[str, ...] = DEFAULT_PERSONAS
    difficulties: tuple[str, ...] = DEFAULT_DIFFICULTIES

    def expand(self, ctx: UnitContext) -> list[SeedPoint]:
        grid = persona_grid(self.roles, self.expertise, self.intents)
        n = min(len(self.personas) * len(self.difficulties), len(grid))
        offset = ctx.rng.randrange(len(grid))
        picked = [grid[(offset + m) % len(grid)] for m in range(n)]
        seeds: list[SeedPoint] = []
        j = 0
        for persona in picked:
            for difficulty in self.difficulties:
                seeds.append(
                    SeedPoint(
                        seed_id=f"{ctx.unit_prefix}:hub:{j}",
                        job_id=ctx.job_id,
                        aspect_id=ctx.aspect_id,
                        topic=ctx.topic,
                        keywords=ctx.keywords,
                        persona=persona,
                        difficulty=difficulty,
                        source_channel=ctx.source_channel,
                        grounding_uri=ctx.grounding_uri,
                        grounding_preview=ctx.grounding_preview,
                        source_url=ctx.source_url,
                        source_license=ctx.source_license,
                        source_sha256=ctx.source_sha256,
                        derived_from=ctx.derived_from,
                    )
                )
                j += 1
        return seeds


def resolve_strategy(
    name: str | None,
    *,
    personas: tuple[str, ...] = DEFAULT_PERSONAS,
    difficulties: tuple[str, ...] = DEFAULT_DIFFICULTIES,
) -> DiversityStrategy:
    """Map a ``diversity_strategy`` value to its implementation.

    Unknown values log a warning and fall back to the default grid so a typo in a
    job spec degrades gracefully instead of failing the run.
    """
    key = (name or "").strip().lower() or DEFAULT_DIVERSITY_STRATEGY
    if key == EVOL_INSTRUCT_STRATEGY:
        return EvolInstructStrategy(personas=personas, difficulties=difficulties)
    if key == PERSONA_HUB_STRATEGY:
        return PersonaHubStrategy(personas=personas, difficulties=difficulties)
    if key != DEFAULT_DIVERSITY_STRATEGY:
        logger.warning(
            "unknown diversity_strategy %r; falling back to %r", name, DEFAULT_DIVERSITY_STRATEGY
        )
    return PersonaGridStrategy(personas=personas, difficulties=difficulties)


# ------------------------------------------------------------------ seed engine
@dataclass
class SeedEngine:
    """Builds a balanced, grounded sampling space from aspects + corpus.

    ``diversity_strategy`` selects how each grounding unit expands (see
    :func:`resolve_strategy`); when unset, ``spec.diversity_strategy`` (passed to
    :meth:`plan`) wins, and the contract default ``persona+evol`` applies last.
    """

    personas: tuple[str, ...] = DEFAULT_PERSONAS
    difficulties: tuple[str, ...] = DEFAULT_DIFFICULTIES
    seed: int = 42
    diversity_strategy: str | None = None

    def plan(
        self,
        job_id: str,
        *,
        aspects: tuple[Aspect, ...],
        chunks: list[CorpusChunk],
        handoffs: list[SynthesisHandoff],
        n_seeds: int,
        spec: JobSpec | None = None,
    ) -> list[SeedPoint]:
        """Return up to ``n_seeds`` seed points, balanced round-robin across aspects."""
        if n_seeds <= 0:
            return []
        seed_value = self.seed
        strategy_name = self.diversity_strategy
        if spec is not None:
            raw_seed = spec.generation_params.get("seed")
            seed_value = self.seed if raw_seed is None else int(raw_seed)
            if strategy_name is None:
                strategy_name = spec.diversity_strategy
        strategy = resolve_strategy(
            strategy_name, personas=self.personas, difficulties=self.difficulties
        )

        by_aspect = {a.aspect_id: a for a in aspects}
        # grounding units per aspect: one per corpus chunk / handoff, or a topic-only
        # unit for aspects with neither
        units: dict[str, list[SeedPoint]] = {}
        ordinals: dict[str, int] = {}

        def _next_ordinal(aid: str) -> int:
            n = ordinals.get(aid, 0)
            ordinals[aid] = n + 1
            return n

        for ch in chunks:
            aspect = by_aspect.get(ch.aspect_id)
            ctx = self._ctx_for_chunk(
                job_id, ch, aspect, rng=random.Random(f"{seed_value}:{ch.aspect_id}:{_next_ordinal(ch.aspect_id)}")
            )
            units.setdefault(ch.aspect_id, []).extend(strategy.expand(ctx))
        for h in handoffs:
            ctx = self._ctx_for_handoff(
                job_id, h, by_aspect, rng=random.Random(f"{seed_value}:{h.aspect_id}:{_next_ordinal(h.aspect_id)}")
            )
            units.setdefault(h.aspect_id, []).extend(strategy.expand(ctx))
        for a in aspects:
            if a.aspect_id not in units:
                ctx = self._ctx_for_aspect(
                    job_id, a, rng=random.Random(f"{seed_value}:{a.aspect_id}:{_next_ordinal(a.aspect_id)}")
                )
                units[a.aspect_id] = strategy.expand(ctx)

        order = sorted(units)  # deterministic aspect order
        for aid in order:
            random.Random(f"{seed_value}:{aid}").shuffle(units[aid])

        out: list[SeedPoint] = []
        idx = {aid: 0 for aid in order}
        while len(out) < n_seeds and any(idx[aid] < len(units[aid]) for aid in order):
            for aid in order:  # round-robin one unit per aspect per pass
                if len(out) >= n_seeds:
                    break
                pool = units[aid]
                if idx[aid] < len(pool):
                    out.append(pool[idx[aid]])
                    idx[aid] += 1
        return out

    # ------------------------------------------------------------------ unit contexts
    def _ctx_for_chunk(
        self, job_id: str, ch: CorpusChunk, aspect: Aspect | None, *, rng: random.Random
    ) -> UnitContext:
        topic = aspect.topic if aspect else ch.aspect_id
        keywords = aspect.keywords if aspect else ()
        return UnitContext(
            unit_prefix=ch.chunk_id,
            job_id=job_id,
            aspect_id=ch.aspect_id,
            topic=topic,
            keywords=keywords,
            source_channel=ch.source_channel,
            grounding_uri=ch.content_uri,
            grounding_preview=ch.text_preview,
            source_url=ch.source_url,
            source_license=ch.source_license,
            source_sha256=ch.source_sha256,
            derived_from=(ch.chunk_id,),
            rng=rng,
        )

    def _ctx_for_handoff(
        self, job_id: str, h: SynthesisHandoff, by_aspect: dict[str, Aspect], *, rng: random.Random
    ) -> UnitContext:
        aspect = by_aspect.get(h.aspect_id)
        keywords = aspect.keywords if aspect else tuple(h.keywords)
        return UnitContext(
            unit_prefix=f"{h.job_id}:{h.aspect_id}:synth",
            job_id=job_id,
            aspect_id=h.aspect_id,
            topic=h.topic,
            keywords=keywords,
            source_channel=SourceChannel.SYNTHESIS,
            rng=rng,
        )

    def _ctx_for_aspect(self, job_id: str, a: Aspect, *, rng: random.Random) -> UnitContext:
        return UnitContext(
            unit_prefix=f"{job_id}:{a.aspect_id}:synth",
            job_id=job_id,
            aspect_id=a.aspect_id,
            topic=a.topic,
            keywords=a.keywords,
            source_channel=SourceChannel.SYNTHESIS,
            rng=rng,
        )


@dataclass
class NoveltyGate:
    """Self-Instruct novelty filter: reject a text too similar to an accepted one.

    Shingle-Jaccard similarity is a deterministic stand-in for ROUGE-L overlap
    (OUTLINE §2.1). Systematic near-duplicate *outputs* are still caught later by the
    MinHash dedup subsystem; this keeps the *sampling space* diverse up front.
    """

    threshold: float = 0.7
    ngram: int = 3
    _seen: list[frozenset] = field(default_factory=list)

    def _shingles(self, text: str) -> frozenset:
        tokens = _WORD_RE.findall((text or "").lower())
        if len(tokens) < self.ngram:
            return frozenset({" ".join(tokens)}) if tokens else frozenset()
        return frozenset(
            " ".join(tokens[i : i + self.ngram]) for i in range(len(tokens) - self.ngram + 1)
        )

    def is_novel(self, text: str) -> bool:
        sh = self._shingles(text)
        if not sh:
            return False
        for prev in self._seen:
            inter = len(sh & prev)
            union = len(sh | prev)
            if union and inter / union >= self.threshold:
                return False
        self._seen.append(sh)
        return True


@dataclass(frozen=True)
class CoverageReport:
    per_aspect: dict[str, int]
    target_per_aspect: float
    thin_aspects: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "per_aspect": self.per_aspect,
            "target_per_aspect": round(self.target_per_aspect, 2),
            "thin_aspects": list(self.thin_aspects),
        }


def coverage_report(
    accepted_aspect_ids: list[str],
    planned_aspects: tuple[Aspect, ...],
    target_records: int,
    *,
    thin_fraction: float = 0.5,
) -> CoverageReport:
    """Count accepted records per aspect and flag aspects below the fair share.

    ``thin_aspects`` are those under ``thin_fraction`` of an even target split — the
    signal the driver uses to request coverage-gap re-discovery (OUTLINE §1.0/§2.1).
    """
    per_aspect: dict[str, int] = {a.aspect_id: 0 for a in planned_aspects}
    for aid in accepted_aspect_ids:
        per_aspect[aid] = per_aspect.get(aid, 0) + 1
    n_aspects = max(len(per_aspect), 1)
    target = target_records / n_aspects
    thin = tuple(aid for aid, n in per_aspect.items() if n < thin_fraction * target)
    return CoverageReport(per_aspect=per_aspect, target_per_aspect=target, thin_aspects=thin)
