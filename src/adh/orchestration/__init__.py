"""Shared spine (minimal) — limits, durable state, saturation, the Engine A driver."""

from __future__ import annotations

from .engine_a import EngineAResult, run_engine_a
from .limits import BreakerTrip, CircuitBreaker, RunMetrics
from .saturation import SaturationTracker
from .state import JobState, load_state, save_state

__all__ = [
    "run_engine_a",
    "EngineAResult",
    "CircuitBreaker",
    "BreakerTrip",
    "RunMetrics",
    "SaturationTracker",
    "JobState",
    "load_state",
    "save_state",
]
