"""Search ports — web meta-search and existing-dataset registries.

Two ports the discovery loop runs queries against:

* :class:`SearchProvider` — web meta-search (SearXNG / Brave / Crawl4AI ``ask``),
  feeding the ``web`` channel.
* :class:`DatasetRegistry` — existing-dataset search (HF Hub / Kaggle / data.gov),
  feeding the ``datasets`` channel — the highest-yield, reuse-first path.

In-memory seedable implementations make discovery testable offline; real adapters
(:mod:`adh.search.hf_registry`) lazy-import the heavy clients.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class SearchHit:
    """One web search result."""

    url: str
    title: str
    snippet: str
    published: str | None = None  # ISO date if known (for recency filtering)
    license_hint: str | None = None


@dataclass(frozen=True)
class DatasetHit:
    """One existing-dataset candidate from a registry."""

    dataset_id: str
    locator: str  # canonical URL, e.g. https://huggingface.co/datasets/<id>
    title: str
    description: str
    license: str | None = None
    downloads: int = 0
    last_modified: str | None = None
    n_examples: int | None = None


@runtime_checkable
class SearchProvider(Protocol):
    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        """Return web results for a query, best-first."""
        ...


@runtime_checkable
class DatasetRegistry(Protocol):
    def search(self, query: str, *, limit: int = 10) -> list[DatasetHit]:
        """Return existing-dataset candidates for a query, best-first."""
        ...

    def sample(self, dataset_id: str, *, max_chars: int = 2000) -> str:
        """Return a short text sample of the dataset for shape evaluation."""
        ...


def _score_overlap(query: str, *texts: str) -> int:
    q = {t for t in query.lower().split() if len(t) > 1}
    blob = " ".join(texts).lower()
    return sum(1 for t in q if t in blob)


@dataclass
class InMemorySearchProvider:
    """Deterministic web search over a seeded page set (offline / tests)."""

    pages: list[SearchHit] = field(default_factory=list)

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        ranked = sorted(
            self.pages,
            key=lambda h: _score_overlap(query, h.title, h.snippet, h.url),
            reverse=True,
        )
        hits = [h for h in ranked if _score_overlap(query, h.title, h.snippet) > 0]
        return hits[:limit]


@dataclass
class InMemoryDatasetRegistry:
    """Deterministic dataset registry over a seeded set (offline / tests)."""

    datasets: list[DatasetHit] = field(default_factory=list)
    samples: dict[str, str] = field(default_factory=dict)

    def search(self, query: str, *, limit: int = 10) -> list[DatasetHit]:
        ranked = sorted(
            self.datasets,
            key=lambda d: (
                _score_overlap(query, d.title, d.description, d.dataset_id),
                d.downloads,
            ),
            reverse=True,
        )
        hits = [d for d in ranked if _score_overlap(query, d.title, d.description) > 0]
        return hits[:limit]

    def sample(self, dataset_id: str, *, max_chars: int = 2000) -> str:
        return self.samples.get(dataset_id, "")[:max_chars]
