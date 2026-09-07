"""Enumerations from the LOCKED Phase 0 contract (PHASE0-CONTRACT.md §0.3.2).

These are the canonical, frozen vocabularies the whole harness is built against.
Do not add members without a corresponding contract change.
"""

from __future__ import annotations

from enum import Enum, IntEnum


class VerifiabilityTier(IntEnum):
    """How correctness is decided — the single most important job parameter.

    The tier selects the verification sub-graph (PHASE0-CONTRACT.md §0.2):
      * HARD  — a deterministic function decides correctness (math/code/schema).
      * SOFT  — no deterministic oracle; quality is graded by an LLM jury.
      * HYBRID— deterministic gate first, LLM jury on the residual dimension.
    """

    HARD = 1
    SOFT = 2
    HYBRID = 3


class DiscoveryMode(str, Enum):
    """Where sources come from. The whole point: ``AUTO`` requires NO sources.

    See PHASE0-CONTRACT.md §0.3 ("Discovery-first").
    """

    AUTO = "auto"  # user gives only the goal; the system discovers sources
    SEEDED = "seeded"  # user gives optional hints; the system expands from them
    FIXED = "fixed"  # user gives exact sources (degenerate; just fetch)


class SourceChannel(str, Enum):
    """Acquisition channels, tried reuse > crawl > acquire > synthesize.

    Phase 0 shipped three members; ``MEDIA`` (licensed stock-video acquisition
    for ``modality="video"`` jobs) was added per the PHASE0-CONTRACT.md §0.3.2
    amendment.
    """

    DATASETS = "datasets"  # HF Hub / Kaggle / data.gov — reuse, don't re-scrape
    WEB = "web"  # Crawl4AI crawl
    MEDIA = "media"  # licensed media download (Pexels / Openverse / open web et al.)
    SYNTHESIS = "synthesis"  # generate from model knowledge, no external source


# Modalities whose jobs acquire real media assets through the MEDIA channel.
# Contract-level vocabulary (PHASE0-CONTRACT.md §0.3.2 media amendment): routing,
# Engine B gating, and the channel itself all branch on this shared definition
# rather than re-hardcoding ``== "video"`` per site.
MEDIA_MODALITIES: frozenset[str] = frozenset({"video", "image"})


def is_media_modality(modality: str) -> bool:
    """True when ``modality`` acquires real assets through the MEDIA channel."""
    return (modality or "").strip().lower() in MEDIA_MODALITIES
