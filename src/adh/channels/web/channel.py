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
from typing import Protocol, runtime_checkable, Callable

from adh.contract import SourceChannel
from adh.corpus import CorpusChunk, offload
from adh.discovery.models import RankedSource

from adh.channels.base import FetchContext, make_chunk_id
from adh.channels.web.internal.service import CrawlService, ServiceConfig
from adh.channels.web.internal.router import CrawlRouter, OnResult
from adh.channels.web.internal.contracts import CrawlRequest, CrawlResponse
from adh.channels.web.internal.relevance import RelevanceJudge
from adh.channels.web.internal.policy import FrontierPolicy, assemble_policy
from adh.provenance import approx_token_count


@dataclass(frozen=True)
class FetchedPage:
    url: str
    markdown: str  # pruned fit_markdown
    license_hint: str | None = None
    relevance_score: float | None = None
    depth: int = 0
    fetched_at: float | None = None
    token_estimate: int | None = None


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
    """Live adapter over Crawl4AI (optional extra ``adh[web]``).

    Owns a singleton :class:`CrawlService` (the browser) and a :class:`CrawlRouter`
    (dispatch + deep-crawl frontier + flush). :meth:`fetch_source` is the rich entry
    the :class:`WebChannel` prefers — it carries the per-aspect :class:`CrawlRequest`
    (constraints + ``aspect_id``) and the saturation ``on_result`` hook. :meth:`fetch`
    is the plain :class:`ContentFetcher` fallback for callers that only have a URL.
    """

    respect_robots_txt: bool = True
    config: ServiceConfig | None = None
    max_depth: int = 2
    # --- relevance / frontier policy (all generic and OFF by default) ---
    # Post-fetch content relevance: a page below ``content_threshold`` (judged against
    # the source's keywords) is dropped and its links not expanded. This is the generic,
    # site-agnostic drift control — pass a Bm25RelevanceJudge / LlmRelevanceJudge.
    judge: RelevanceJudge | None = None
    content_threshold: float = 0.0
    # Pre-fetch URL/anchor keyword scoring (cheap prioritisation; brittle on sites that
    # don't encode topic in the path). Off by default.
    score_urls_by_keyword: bool = False
    url_threshold: float = 0.0
    # Opt-in, site-specific dedup/filters (NOT defaults — see policy.py):
    collapse_versions: bool = False  # versioned-docs dedup (/en/6.0/x == /en/5.1/x)
    pin_locales: bool = False  # i18n sites: keep only the request's allowed_locales
    # Optional injection points (mirrors CrawlService) so the live adapter and its
    # router/frontier can be exercised offline with a fake crawler — no browser.
    crawler_factory: object | None = None
    run_config_factory: object | None = None
    _service: CrawlService | None = field(default=None, init=False, repr=False)
    _router: CrawlRouter | None = field(default=None, init=False, repr=False)

    def fetch_source(
        self,
        request: CrawlRequest,
        *,
        on_result: OnResult | None = None,
        policy: FrontierPolicy | None = None,
        max_fetches: int | None = None,
        max_depth: int | None = None,
        on_event: Callable[[str, str, dict], None] | None = None,
    ) -> list[FetchedPage]:
        """Deep-crawl a seed request through the router.

        By default the policy is assembled from this fetcher's configuration; a caller
        that needs per-job steering (the crawl microservice) passes ``policy`` — and
        optionally ``max_fetches`` (bound fetched, not just kept, pages) and
        ``max_depth`` (override this fetcher's default depth) — directly.
        """
        router = self._ensure_router()
        responses = router.run_job(
            request,
            max_pages=request.max_pages,
            max_fetches=max_fetches,
            max_depth=max_depth,
            on_result=on_result,
            policy=policy or self._build_policy(request),
            on_event=on_event,
        )
        return [_response_to_page(r) for r in responses]

    def fetch(self, url: str, *, max_pages: int = 1) -> list[FetchedPage]:
        request = CrawlRequest(
            url=url, max_pages=max_pages, respect_robots_txt=self.respect_robots_txt
        )
        return self.fetch_source(request)

    def _build_policy(self, request: CrawlRequest) -> FrontierPolicy:
        """Assemble the frontier policy for one seed from this fetcher's configuration."""
        return assemble_policy(
            request,
            judge=self.judge,
            content_threshold=self.content_threshold,
            score_urls_by_keyword=self.score_urls_by_keyword,
            url_threshold=self.url_threshold,
            collapse_versions=self.collapse_versions,
            pin_locales=self.pin_locales,
        )

    def cancel_aspect(self, aspect_id: str) -> int:
        """Flush every in-flight deep-crawl job serving ``aspect_id`` (thread-safe).

        The microservice stamps its job id as the seed's ``aspect_id``, so this is
        also its per-job cancel. Returns how many router jobs were flushed.
        """
        return self._router.flush_aspect(aspect_id) if self._router is not None else 0

    def _ensure_router(self) -> CrawlRouter:
        if self._router is None:
            service = self._ensure_service()
            self._router = CrawlRouter(
                service, workers=service.config.max_concurrent_pages, max_depth=self.max_depth
            )
        return self._router

    def _ensure_service(self) -> CrawlService:
        if self._service is None:
            kwargs = {}
            if self.crawler_factory is not None:
                kwargs["crawler_factory"] = self.crawler_factory
            if self.run_config_factory is not None:
                kwargs["run_config_factory"] = self.run_config_factory
            self._service = CrawlService(self.config, **kwargs)
        return self._service

    def close(self) -> None:
        if self._service is not None:
            self._service.close()
        self._service = None
        self._router = None


def _response_to_page(resp: CrawlResponse) -> FetchedPage:
    """Map a :class:`CrawlResponse` to a :class:`FetchedPage`.

    Provenance records where the content *actually* came from: the post-redirect
    ``final_url`` when present, falling back to the requested URL.
    """
    return FetchedPage(
        url=resp.final_url or resp.url,
        markdown=resp.markdown,
        license_hint=resp.license_hint,
        relevance_score=resp.relevance_score,
        depth=resp.depth,
        fetched_at=resp.fetched_at,
        token_estimate=approx_token_count(resp.markdown) if resp.markdown else 0,
    )


def _build_request(source: RankedSource, ctx: FetchContext) -> CrawlRequest:
    """Translate one approved source + the JobSpec constraints into a seed request."""
    c = ctx.spec.discovery.constraints
    return CrawlRequest(
        url=source.candidate.locator,
        max_pages=ctx.max_items_per_source,
        aspect_id=source.candidate.aspect_id,
        respect_robots_txt=c.respect_robots_txt,
        include_domains=tuple(c.include_domains) or None,
        exclude_domains=tuple(c.exclude_domains) or None,
        keywords=_relevance_keywords(source),
        allowed_locales=tuple(ctx.spec.languages),
    )


def _relevance_keywords(source: RankedSource) -> tuple[str, ...]:
    """The aspect vocabulary that steers the deep-crawl frontier.

    Discovery's per-source signal — title, aspect id, and snippet — is exactly what
    "this source is about" means, so we reuse it to score outlink relevance. (When
    discovery attaches richer ``Aspect.keywords`` upstream, they flow in here too.)
    """
    cand = source.candidate
    aspect_words = cand.aspect_id.replace("-", " ").replace("_", " ")
    return tuple(p for p in (cand.title, aspect_words, cand.snippet) if p)


def _saturation_hook(governor, aspect_id: str) -> OnResult:
    """Per-page hook: feed the coverage Brain, then ask whether to keep crawling."""

    def hook(resp: CrawlResponse) -> bool:
        governor.observe(aspect_id, resp.markdown)
        return not governor.should_stop(aspect_id)

    return hook


@dataclass
class WebChannel:
    """Acquire web content for an approved source and offload it as corpus chunks."""

    fetcher: ContentFetcher
    name: SourceChannel = SourceChannel.WEB

    def fetch(self, source: RankedSource, ctx: FetchContext) -> list[CorpusChunk]:
        aspect_id = source.candidate.aspect_id
        pages = self._acquire(source, ctx, aspect_id)
        chunks: list[CorpusChunk] = []
        for page in pages[: ctx.max_items_per_source]:
            if not page.markdown.strip():
                continue
            license_ = source.candidate.license or page.license_hint or "unknown"
            chunks.append(
                offload(
                    ctx.store,
                    job_id=ctx.job_id,
                    aspect_id=aspect_id,
                    content=page.markdown,
                    source_channel=SourceChannel.WEB,
                    source_url=page.url,
                    source_license=license_,
                    chunk_id=make_chunk_id(ctx.job_id, aspect_id, page.markdown),
                    discovered_via=source.candidate.discovered_via,
                    fetched_at=ctx.clock(),
                )
            )
        return chunks

    def _acquire(
        self, source: RankedSource, ctx: FetchContext, aspect_id: str
    ) -> list[FetchedPage]:
        """Fetch pages, threading constraints + the saturation Brain to the fetcher.

        A constraint-aware fetcher (the live Crawl4AI adapter) gets the full request
        and an ``on_result`` flush hook driven by the governor. A plain
        :class:`ContentFetcher` only takes a URL, so we report coverage to the
        governor *after* the (single-shot) fetch instead.
        """
        fetcher = self.fetcher
        if hasattr(fetcher, "fetch_source"):
            seed = _build_request(source, ctx)
            return fetcher.fetch_source(seed, on_result=_saturation_hook(ctx.governor, aspect_id))

        pages = fetcher.fetch(source.candidate.locator, max_pages=ctx.max_items_per_source)
        for page in pages[: ctx.max_items_per_source]:
            if page.markdown.strip():
                ctx.governor.observe(aspect_id, page.markdown)
        return pages

    def close(self) -> None:
        if hasattr(self.fetcher, "close"):
            self.fetcher.close()
