"""Seed / taxonomy engine (OUTLINE §2.1) — diversity is the systemic risk.

Turns the discovered aspects + acquired corpus into a *structured sampling space*
rather than chasing raw volume: every seed crosses an aspect's grounding with a
persona and a difficulty, distributed round-robin across aspects so no single topic
dominates. A :class:`NoveltyGate` (shingle-Jaccard, a deterministic ROUGE-L proxy)
implements the Self-Instruct novelty filter, and :func:`coverage_report` tracks how
the accepted set covers the taxonomy so the driver can ask Engine A to re-discover
thin aspects (the coverage-gap loop).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field

from adh.corpus import CorpusChunk, SynthesisHandoff
from adh.discovery.models import Aspect

from .models import DEFAULT_DIFFICULTIES, DEFAULT_PERSONAS, SeedPoint

_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass
class SeedEngine:
    """Builds a balanced, grounded sampling space from aspects + corpus."""

    personas: tuple[str, ...] = DEFAULT_PERSONAS
    difficulties: tuple[str, ...] = DEFAULT_DIFFICULTIES
    seed: int = 42

    def plan(
        self,
        job_id: str,
        *,
        aspects: tuple[Aspect, ...],
        chunks: list[CorpusChunk],
        handoffs: list[SynthesisHandoff],
        n_seeds: int,
    ) -> list[SeedPoint]:
        """Return up to ``n_seeds`` seed points, balanced round-robin across aspects."""
        if n_seeds <= 0:
            return []
        by_aspect = {a.aspect_id: a for a in aspects}
        # grounding units per aspect: one per corpus chunk, or a topic-only unit for synthesis
        units: dict[str, list[SeedPoint]] = {}
        for ch in chunks:
            aspect = by_aspect.get(ch.aspect_id)
            units.setdefault(ch.aspect_id, []).extend(
                self._units_for_chunk(job_id, ch, aspect)
            )
        for h in handoffs:
            units.setdefault(h.aspect_id, []).extend(self._units_for_handoff(job_id, h, by_aspect))
        # aspects with neither corpus nor handoff still deserve synthesis coverage
        for a in aspects:
            if a.aspect_id not in units:
                units[a.aspect_id] = self._units_for_aspect(job_id, a)

        order = sorted(units)  # deterministic aspect order
        for aid in order:
            random.Random(f"{self.seed}:{aid}").shuffle(units[aid])

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

    # ------------------------------------------------------------------ builders
    def _combos(self):
        for p in self.personas:
            for d in self.difficulties:
                yield p, d

    def _units_for_chunk(self, job_id, ch: CorpusChunk, aspect: Aspect | None) -> list[SeedPoint]:
        topic = aspect.topic if aspect else ch.aspect_id
        keywords = aspect.keywords if aspect else ()
        seeds = []
        for i, (persona, difficulty) in enumerate(self._combos()):
            seeds.append(
                SeedPoint(
                    seed_id=f"{ch.chunk_id}:{i}",
                    job_id=job_id,
                    aspect_id=ch.aspect_id,
                    topic=topic,
                    keywords=keywords,
                    persona=persona,
                    difficulty=difficulty,
                    source_channel=ch.source_channel,
                    grounding_uri=ch.content_uri,
                    grounding_preview=ch.text_preview,
                    source_url=ch.source_url,
                    source_license=ch.source_license,
                    source_sha256=ch.source_sha256,
                    derived_from=(ch.chunk_id,),
                )
            )
        return seeds

    def _units_for_handoff(self, job_id, h: SynthesisHandoff, by_aspect) -> list[SeedPoint]:
        from adh.contract import SourceChannel

        aspect = by_aspect.get(h.aspect_id)
        keywords = aspect.keywords if aspect else tuple(h.keywords)
        return [
            SeedPoint(
                seed_id=f"{h.job_id}:{h.aspect_id}:synth:{i}",
                job_id=job_id,
                aspect_id=h.aspect_id,
                topic=h.topic,
                keywords=keywords,
                persona=persona,
                difficulty=difficulty,
                source_channel=SourceChannel.SYNTHESIS,
            )
            for i, (persona, difficulty) in enumerate(self._combos())
        ]

    def _units_for_aspect(self, job_id, a: Aspect) -> list[SeedPoint]:
        from adh.contract import SourceChannel

        return [
            SeedPoint(
                seed_id=f"{job_id}:{a.aspect_id}:synth:{i}",
                job_id=job_id,
                aspect_id=a.aspect_id,
                topic=a.topic,
                keywords=a.keywords,
                persona=persona,
                difficulty=difficulty,
                source_channel=SourceChannel.SYNTHESIS,
            )
            for i, (persona, difficulty) in enumerate(self._combos())
        ]


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
