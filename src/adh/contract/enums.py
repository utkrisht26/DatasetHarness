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
    """The three Phase 0 acquisition channels, tried reuse > crawl > synthesize."""

    DATASETS = "datasets"  # HF Hub / Kaggle / data.gov — reuse, don't re-scrape
    WEB = "web"  # Crawl4AI crawl
    SYNTHESIS = "synthesis"  # generate from model knowledge, no external source
