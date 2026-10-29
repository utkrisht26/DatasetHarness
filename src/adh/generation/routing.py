"""Cost & model routing for generation (OUTLINE §3.2).

The ``generator_models`` ladder is applied at data scale, mirroring the /model-route
heuristic: *bulk* generation runs on the cheapest configured tier while *hard cases*
escalate to the frontier tier — where "hard" is decided twice:

1. **up front** by seed difficulty — ``advanced`` seeds start at the last tier
   (frontier models first; they are not wasted on retries of a doomed cheap call);
2. **reactively** on failure — a tier that raises persistently (after bounded retries
   for transient provider faults) or returns no parseable candidates hands the seed
   to the next tier up.

Every dispatch is observable: :attr:`RoutedGenerator.calls` counts seeds per tier,
:attr:`~RoutedGenerator.escalations` counts reactive upgrades and
:attr:`~RoutedGenerator.exhausted` counts seeds no tier could serve (the signal to
revisit the model ladder or the prompts).

Cost itself stays decision-4 honest: spend is metered per call into the run's
:class:`~adh.llm.UsageTracker` from provider-surfaced usage only — no price table.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable

from .generator import Generator, LlmGenerator, SeedSkipped

logger = logging.getLogger(__name__)

# Seed difficulties that skip the cheap tiers entirely (OUTLINE §3.2 "hard cases").
_FRONTIER_DIFFICULTIES = frozenset({"advanced"})

# Transient provider faults (timeouts, 5xx blips) retry within their tier before
# escalation — a single-tier ladder otherwise has zero tolerance, and one bad upstream
# minute can silently zero out an entire build. Empty-but-clean results are NOT
# retried: that is a quality verdict, not a fault.
_TIER_ATTEMPTS = 3
_RETRY_BACKOFF_SECONDS = (0.5, 1.0)
_sleep = time.sleep  # indirection so tests can fast-forward the backoff


@dataclass
class RoutedGenerator:
    """Dispatches each seed across an ordered cheap->frontier tier ladder."""

    tiers: tuple[Callable[[], Generator], ...]
    _built: dict[int, Generator] = field(default_factory=dict, init=False)
    _calls: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int), init=False)
    escalations: int = field(default=0, init=False)
    exhausted: int = field(default=0, init=False)

    @property
    def calls(self) -> dict[str, int]:
        """Seeds dispatched per tier label (``tier0``, ``tier1``, …)."""
        return dict(self._calls)

    def generate(self, seed, *, spec, store=None) -> list:
        start = len(self.tiers) - 1 if seed.difficulty in _FRONTIER_DIFFICULTIES else 0
        for index in range(start, len(self.tiers)):
            self._calls[f"tier{index}"] += 1
            candidates = self._dispatch(index, seed, spec=spec, store=store)
            if candidates:
                return candidates
            if candidates is None:
                logger.warning(
                    "generation tier%d exhausted %d attempt(s) for seed %s",
                    index,
                    _TIER_ATTEMPTS,
                    seed.seed_id,
                )
            else:
                logger.info("generation tier%d returned nothing for seed %s", index, seed.seed_id)
            if index + 1 < len(self.tiers):
                self.escalations += 1
        self.exhausted += 1
        return []

    def _dispatch(self, index: int, seed, *, spec, store) -> list | None:
        """One tier's dispatch with bounded retries; ``None`` = every attempt raised."""
        generator = self._tier(index)
        for attempt in range(_TIER_ATTEMPTS):
            try:
                return generator.generate(seed, spec=spec, store=store)
            except SeedSkipped:
                raise  # a verdict on the grounding: every tier would see the same junk
            except Exception as exc:  # noqa: BLE001 - a failed attempt retries, then escalates
                logger.warning(
                    "generation tier%d attempt %d/%d failed on seed %s: %s",
                    index,
                    attempt + 1,
                    _TIER_ATTEMPTS,
                    seed.seed_id,
                    exc,
                )
                if attempt < _TIER_ATTEMPTS - 1 and _RETRY_BACKOFF_SECONDS:
                    delay = _RETRY_BACKOFF_SECONDS[min(attempt, len(_RETRY_BACKOFF_SECONDS) - 1)]
                    _sleep(delay)
        return None

    def _tier(self, index: int) -> Generator:
        """Resolve a tier once; entries may be ready generators or lazy factories."""
        if index not in self._built:
            component = self.tiers[index]
            self._built[index] = component() if callable(component) else component
        return self._built[index]


def routed_from_models(model_ids, usage=None) -> tuple[Callable[[], LlmGenerator], ...]:
    """Lazy per-model constructors so an unused frontier tier never builds."""
    return tuple(lambda mid=mid, u=usage: LlmGenerator(model_id=mid, usage=u) for mid in model_ids)
