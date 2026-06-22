"""The ``synthesis`` channel — no corpus, straight hand-off to Engine B (OUTLINE §1.7).

When discovery routes an aspect to synthesis, there is no acquisition: the aspect flows
directly to Engine B generation (Phase 2.2), seeded by the goal/taxonomy. Records emitted
later stamp ``source_channel="synthesis"`` and ``source_url=null``. Here we only record
the hand-off so the manifest and lineage stay complete.
"""

from __future__ import annotations

from adh.corpus import SynthesisHandoff
from adh.discovery.models import Aspect


def handoffs_for(job_id: str, aspects: tuple[Aspect, ...]) -> list[SynthesisHandoff]:
    """Build synthesis hand-off records for the synthesis-routed aspects."""
    return [
        SynthesisHandoff(
            job_id=job_id,
            aspect_id=a.aspect_id,
            topic=a.topic,
            keywords=a.keywords,
        )
        for a in aspects
    ]
