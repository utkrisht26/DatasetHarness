"""Limits & circuit breaker — the safety floor (OUTLINE §1.4, §3.1).

Coverage-saturation is the *primary* stop signal; these hard limits are the
circuit-breaker floor that guarantees a run terminates. :class:`RunMetrics` is an
immutable accumulator (LangGraph-style reducers): every update returns a new value.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from adh.contract import Limits


@dataclass(frozen=True)
class RunMetrics:
    """Accumulated run metrics. Immutable — ``add_*`` return new instances.

    ``cost_usd`` is ``None`` until a provider actually surfaces cost (decision 4:
    cost is never derived from a self-maintained price table — it is real or
    reported unknown).
    """

    pages_scraped: int = 0
    llm_tokens: int = 0
    cost_usd: float | None = None

    def add_pages(self, n: int) -> "RunMetrics":
        return replace(self, pages_scraped=self.pages_scraped + n)

    def add_tokens(self, n: int) -> "RunMetrics":
        return replace(self, llm_tokens=self.llm_tokens + n)

    def add_cost(self, usd: float | None) -> "RunMetrics":
        if usd is None:
            return self  # provider surfaced no cost -> stays unknown, never guessed
        return replace(self, cost_usd=round((self.cost_usd or 0.0) + usd, 6))

    def to_dict(self) -> dict:
        return {
            "pages_scraped": self.pages_scraped,
            "llm_tokens": self.llm_tokens,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunMetrics":
        cost = data.get("cost_usd")
        return cls(
            pages_scraped=int(data.get("pages_scraped", 0)),
            llm_tokens=int(data.get("llm_tokens", 0)),
            cost_usd=float(cost) if cost is not None else None,
        )


@dataclass(frozen=True)
class BreakerTrip:
    """A tripped limit: which one, and the offending value vs the cap."""

    reason: str
    value: float
    limit: float

    def __str__(self) -> str:
        return f"limit '{self.reason}' reached: {self.value} >= {self.limit}"


@dataclass
class CircuitBreaker:
    """Two-layer floor: consulted before each fetch and after metrics update."""

    limits: Limits

    def check(
        self,
        metrics: RunMetrics,
        *,
        storage_mb: float = 0.0,
        elapsed_seconds: float = 0.0,
    ) -> BreakerTrip | None:
        """Return the first tripped limit, or ``None`` if all are within bounds.

        This is a pre-fetch gate, so it trips once a metric has *reached* its cap
        (``>=``): the floor guarantees the run never goes over the limit, and a cap
        of ``0`` means "do not start that work at all".
        """
        checks: list[tuple[str, float, float]] = [
            ("max_pages_scraped", metrics.pages_scraped, self.limits.max_pages_scraped),
            ("max_llm_tokens", metrics.llm_tokens, self.limits.max_llm_tokens),
        ]
        if metrics.cost_usd is not None:  # unknown cost is unenforceable (decision 4)
            checks.append(("max_cost_usd", metrics.cost_usd, self.limits.max_cost_usd))
        checks += [
            ("max_storage_mb", storage_mb, self.limits.max_storage_mb),
            ("max_runtime_seconds", elapsed_seconds, self.limits.max_runtime_seconds),
        ]
        for reason, value, limit in checks:
            if value >= limit:
                return BreakerTrip(reason=reason, value=float(value), limit=float(limit))
        return None
