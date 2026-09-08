"""Tavily-backed :class:`SearchProvider` — live web meta-search for discovery.

The in-memory stub returns nothing for real-world queries, which sends the
discovery agent into dead-end searches that burn token budget without producing
candidates (the failure mode of every pre-Tavily live run). This adapter wires
the same :class:`SearchProvider` port to the Tavily REST API so ``web_search``
tool results carry real candidates.

Failure policy: a search error degrades to an empty result list with a warning
log — discovery must keep moving; the agent falls back to model-knowledge seeds
(the behaviour the contract pins for "no search results").

The client is injected via ``client_factory`` so tests run hermetically; the
default lazily imports the optional ``tavily`` package.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Callable

from .provider import SearchHit

logger = logging.getLogger(__name__)

# Snippets are steering hints for the crawler frontier, not corpus content —
# cap them so one bloated result cannot dominate the search-result file.
_MAX_SNIPPET_CHARS = 400


class TavilySearchProvider:
    """Live meta-search over the Tavily API behind the ``SearchProvider`` port."""

    def __init__(
        self,
        api_key: str | None = None,
        client_factory: Callable[[], object] | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else os.environ.get("TAVILY_API_KEY", "")
        # An injected factory stands in for the whole client (tests, proxies), so
        # searches work through it even with no local key configured.
        self._client_factory = client_factory or self._build_default_client
        self._client: object | None = None
        self._client_lock = threading.Lock()

    # ------------------------------------------------------------- port API
    @property
    def available(self) -> bool:
        """Whether a Tavily API key is configured."""
        return bool(self.api_key)

    def search(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        """Query Tavily and map results to :class:`SearchHit`; errors -> [].

        A missing key/package or a failed request logs a warning and returns an
        empty list — search failure must never kill discovery (the agent falls
        back to model-knowledge seeds).
        """
        try:
            response = self._tavily().search(query, max_results=limit)
        except Exception as exc:  # noqa: BLE001 - degrade, never kill discovery
            logger.warning("tavily search failed for query %r: %s", query, exc)
            return []
        return [hit for hit in map(self._to_hit, _results(response)) if hit is not None]

    # ---------------------------------------------------------------- internals
    def _build_default_client(self) -> object:
        if not self.api_key:
            raise RuntimeError("TAVILY_API_KEY is not set; configure it or inject a client_factory")
        try:
            from tavily import TavilyClient
        except ImportError as exc:  # pragma: no cover - needs the search extra
            raise RuntimeError(
                "Tavily search requires the tavily package; install with: pip install 'adh[search]'"
            ) from exc
        return TavilyClient(api_key=self.api_key)

    def _tavily(self) -> object:
        """Memoized client; the factory runs once, under a lock."""
        with self._client_lock:
            if self._client is None:
                self._client = self._client_factory()
            return self._client

    @staticmethod
    def _to_hit(result: dict) -> SearchHit | None:
        url = result.get("url")
        if not url:
            return None
        snippet = " ".join(str(result.get("content") or "").split())
        return SearchHit(
            url=str(url),
            title=str(result.get("title") or url),
            snippet=snippet[:_MAX_SNIPPET_CHARS],
            published=result.get("published_date"),
        )


def _results(response: object) -> list[dict]:
    """Extract the result dicts from a Tavily response (dict- or object-shaped)."""
    raw = getattr(response, "results", None)
    if raw is None and isinstance(response, dict):
        raw = response.get("results")
    return [r for r in (raw or []) if isinstance(r, dict)]
