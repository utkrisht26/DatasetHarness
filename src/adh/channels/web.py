"""The ``web`` channel — Crawl4AI crawl (OUTLINE §1.1).

A :class:`ContentFetcher` port abstracts the actual crawl so the channel runs offline
with :class:`InMemoryContentFetcher` and in production with
:class:`Crawl4AIContentFetcher` (the official Crawl4AI MCP server / ``AsyncWebCrawler``).
The fetcher returns already-pruned ``fit_markdown`` (Crawl4AI ``PruningContentFilter`` /
``BM25ContentFilter``) so heavy raw HTML never crosses the boundary (OUTLINE §1.3).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from adh.contract import SourceChannel
from adh.corpus import CorpusChunk, offload
from adh.discovery.models import RankedSource

from .base import FetchContext, make_chunk_id


@dataclass(frozen=True)
class FetchedPage:
    url: str
    markdown: str  # pruned fit_markdown
    license_hint: str | None = None


@runtime_checkable
class ContentFetcher(Protocol):
    def fetch(self, url: str, *, max_pages: int = 1) -> list[FetchedPage]:
        """Fetch a URL (optionally deep-crawling up to ``max_pages``)."""
        ...


@dataclass
class InMemoryContentFetcher:
    """Deterministic fetcher over a seeded page set, with optional deep-crawl links."""

    pages: dict[str, str] = field(default_factory=dict)  # url -> markdown
    links: dict[str, list[str]] = field(default_factory=dict)  # url -> outlinks
    license_hints: dict[str, str] = field(default_factory=dict)

    def fetch(self, url: str, *, max_pages: int = 1) -> list[FetchedPage]:
        out: list[FetchedPage] = []
        seen: set[str] = set()
        queue: deque[str] = deque([url])
        while queue and len(out) < max_pages:
            current = queue.popleft()
            if current in seen:
                continue
            seen.add(current)
            md = self.pages.get(current)
            if md is not None:
                out.append(
                    FetchedPage(url=current, markdown=md, license_hint=self.license_hints.get(current))
                )
            for nxt in self.links.get(current, []):
                if nxt not in seen:
                    queue.append(nxt)
        return out


@dataclass
class Crawl4AIContentFetcher:
    """Live adapter over Crawl4AI (optional extra ``adh[web]``), lazy-imported.

    Uses ``BestFirstCrawlingStrategy`` + ``PruningContentFilter`` and returns
    ``result.markdown.fit_markdown``. ``respect_robots_txt`` and proxy rotation are
    delegated to Crawl4AI's config (RESEARCH §APIs).
    """

    respect_robots_txt: bool = True

    def fetch(self, url: str, *, max_pages: int = 1) -> list[FetchedPage]:  # pragma: no cover
        import asyncio

        return asyncio.run(self._afetch(url, max_pages=max_pages))

    async def _afetch(self, url: str, *, max_pages: int) -> list[FetchedPage]:  # pragma: no cover
        try:
            from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig
            from crawl4ai.content_filter_strategy import PruningContentFilter
        except ImportError as exc:
            raise RuntimeError(
                "crawl4ai is required for the live web channel; install with: pip install 'adh[web]'"
            ) from exc

        run_cfg = CrawlerRunConfig(
            check_robots_txt=self.respect_robots_txt,
            content_filter=PruningContentFilter(threshold=0.48),
        )
        out: list[FetchedPage] = []
        async with AsyncWebCrawler(config=BrowserConfig()) as crawler:
            result = await crawler.arun(url, config=run_cfg)
            md = getattr(getattr(result, "markdown", None), "fit_markdown", "") or ""
            if md:
                out.append(FetchedPage(url=url, markdown=md))
        return out[:max_pages]


@dataclass
class WebChannel:
    """Acquire web content for an approved source and offload it as corpus chunks."""

    fetcher: ContentFetcher
    name: SourceChannel = SourceChannel.WEB

    def fetch(self, source: RankedSource, ctx: FetchContext) -> list[CorpusChunk]:
        pages = self.fetcher.fetch(
            source.candidate.locator, max_pages=ctx.max_items_per_source
        )
        chunks: list[CorpusChunk] = []
        for page in pages[: ctx.max_items_per_source]:
            if not page.markdown.strip():
                continue
            license_ = source.candidate.license or page.license_hint or "unknown"
            chunks.append(
                offload(
                    ctx.store,
                    job_id=ctx.job_id,
                    aspect_id=source.candidate.aspect_id,
                    content=page.markdown,
                    source_channel=SourceChannel.WEB,
                    source_url=page.url,
                    source_license=license_,
                    chunk_id=make_chunk_id(ctx.job_id, source.candidate.aspect_id, page.markdown),
                    discovered_via=source.candidate.discovered_via,
                    fetched_at=ctx.clock(),
                )
            )
        return chunks
