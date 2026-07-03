"""Hardened Crawl4AI service bridge (T2).

A long-lived async event loop in a background thread exposing a *synchronous*
`crawl()` to the (synchronous) `WebChannel`, backed by a singleton browser. The
design from the B1 spike, hardened per the engineering review:

* background-loop singleton + sync entry via ``run_coroutine_threadsafe``
* ``asyncio.Lock`` around init so concurrent cold-starts don't double-launch
* timeout-bounded so the sync caller never hangs
* recovery on *fatal* (browser-dead) errors only — a single slow/bad URL does not
  tear down the shared browser; the old crawler is closed before re-init (no leak)
* a configurable concurrency semaphore (OOM floor)
* fetch-time enforcement of robots.txt, domain allow/deny, and a basic SSRF guard
* idempotent ``close()``

Crawl4AI is an **optional extra** (``adh[web]``): it is imported lazily inside the
default factory, so this module imports and unit-tests fine offline with an injected
fake crawler.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import threading
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from dataclasses import dataclass
from typing import Callable, Optional, Protocol
from urllib.parse import urlparse

from .contracts import CrawlRequest, CrawlResponse, Outlink

logger = logging.getLogger(__name__)


def _echo(request: CrawlRequest) -> dict:
    """Response kwargs mirrored from the request so the router can route results."""
    return {"aspect_id": request.aspect_id, "depth": request.depth}


def _extract_links(result: object) -> tuple[Outlink, ...]:
    """Best-effort outlink extraction (href + anchor text) for the frontier.

    Crawl4AI exposes ``result.links`` as ``{"internal": [...], "external": [...]}``
    where each entry is a dict carrying ``href`` (and usually ``text``/``title``).
    Anchor text feeds the relevance scorer. Stay defensive: anything we can't parse
    yields no links rather than raising (a bad links payload must not fail an
    otherwise-good crawl).
    """
    links = getattr(result, "links", None)
    if not links:
        return ()
    groups = links.values() if isinstance(links, dict) else [links]
    out: list[Outlink] = []
    seen: set[str] = set()
    for group in groups:
        if not isinstance(group, (list, tuple)):
            continue
        for item in group:
            href: str | None = None
            text = ""
            if isinstance(item, str):
                href = item
            elif isinstance(item, dict):
                href = item.get("href") or item.get("url")
                text = item.get("text") or item.get("title") or ""
            if isinstance(href, str) and href not in seen:
                seen.add(href)
                out.append(Outlink(url=href, text=text if isinstance(text, str) else ""))
    return tuple(out)  # dedupe by href, preserve discovery order


class _Crawler(Protocol):
    """The slice of Crawl4AI's AsyncWebCrawler the service depends on."""

    async def start(self) -> object: ...
    async def close(self) -> object: ...
    async def arun(self, url: str, config: object | None = None) -> object: ...


# Factories let tests inject fakes; the defaults lazy-import crawl4ai.
CrawlerFactory = Callable[[], _Crawler]
RunConfigFactory = Callable[[CrawlRequest], object | None]


class Crawl4AIUnavailableError(RuntimeError):
    """Raised when the live web channel is used without the ``adh[web]`` extra."""


def _default_crawler_factory() -> _Crawler:
    try:
        from crawl4ai import AsyncWebCrawler, BrowserConfig
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise Crawl4AIUnavailableError(
            "crawl4ai is required for the live web channel; install with: pip install 'adh[web]'"
        ) from exc
    return AsyncWebCrawler(config=BrowserConfig())


def _default_run_config(request: CrawlRequest) -> object:  # pragma: no cover - needs crawl4ai
    # crawl4ai 0.9 moved the content filter behind the markdown generator;
    # ``CrawlerRunConfig(content_filter=...)`` (the 0.8 signature) is gone. The
    # PruningContentFilter still produces ``result.markdown.fit_markdown``.
    from crawl4ai import CrawlerRunConfig
    from crawl4ai.content_filter_strategy import PruningContentFilter
    from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator

    return CrawlerRunConfig(
        check_robots_txt=request.respect_robots_txt,
        markdown_generator=DefaultMarkdownGenerator(
            content_filter=PruningContentFilter(threshold=0.48)
        ),
    )


@dataclass
class ServiceConfig:
    """Tunables. Defaults are hardware-safe for a typical laptop."""

    max_concurrent_pages: int = 4
    request_timeout: float = 60.0
    browser_init_timeout: float = 30.0
    # Service-level safety cap on total pages, distinct from (and complementary to)
    # the orchestrator's per-job circuit breaker. A backstop, not the primary bound.
    max_pages_scraped: int = 1000
    block_private_addresses: bool = True


class CrawlService:
    """Thread-safe synchronous bridge to a singleton async browser."""

    def __init__(
        self,
        config: ServiceConfig | None = None,
        *,
        crawler_factory: CrawlerFactory | None = None,
        run_config_factory: RunConfigFactory | None = None,
    ) -> None:
        self.config = config or ServiceConfig()
        self._crawler_factory = crawler_factory or _default_crawler_factory
        self._run_config_factory = run_config_factory or _default_run_config

        self._loop = asyncio.new_event_loop()
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_pages)
        self._crawler: Optional[_Crawler] = None
        self._pages_scraped = 0
        self._is_closing = False
        self._closed = False

        self._thread = threading.Thread(
            target=self._run_event_loop, name="crawl-service-loop", daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------ loop
    def _run_event_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    # ------------------------------------------------------------------ crawler lifecycle
    async def _ensure_crawler(self) -> _Crawler:
        async with self._lock:
            if self._crawler is not None:
                return self._crawler
            logger.info("Initializing crawler singleton...")
            crawler = self._crawler_factory()
            await asyncio.wait_for(crawler.start(), timeout=self.config.browser_init_timeout)
            self._crawler = crawler
            return crawler

    async def _recover(self) -> None:
        """Tear down a (presumed dead) browser so the next request re-inits it.

        Closes the old instance first to avoid leaking a browser process.
        """
        async with self._lock:
            old, self._crawler = self._crawler, None
        if old is not None:
            try:
                await old.close()
            except Exception as exc:  # noqa: BLE001 - best-effort cleanup
                logger.warning("Error closing crawler during recovery: %s", exc)

    # ------------------------------------------------------------------ guards
    def _reject_reason(self, request: CrawlRequest) -> str | None:
        """Return why a URL must not be fetched, or ``None`` if allowed.

        Enforces scheme (SSRF), host presence, domain allow/deny, and (optionally)
        blocks private / loopback / link-local addresses (cloud-metadata SSRF).
        """
        parsed = urlparse(request.url)
        if parsed.scheme not in ("http", "https"):
            return f"blocked non-http(s) scheme: {parsed.scheme or 'none'!r}"
        host = (parsed.hostname or "").lower()
        if not host:
            return "blocked: URL has no host"

        excl = request.exclude_domains or ()
        if any(host == d.lower() or host.endswith("." + d.lower()) for d in excl):
            return f"blocked by exclude_domains: {host}"
        incl = request.include_domains or ()
        if incl and not any(host == d.lower() or host.endswith("." + d.lower()) for d in incl):
            return f"not in include_domains: {host}"

        if self.config.block_private_addresses and self._is_private_host(host):
            return f"blocked private/internal address (SSRF guard): {host}"
        return None

    @staticmethod
    def _is_private_host(host: str) -> bool:
        """Block localhost + literal private/loopback/link-local IPs.

        Intentionally does NOT resolve hostnames: a synchronous ``getaddrinfo`` here
        would block the event loop and do real DNS. Catching a hostname that *resolves*
        to a private IP (DNS-rebinding) needs async resolution and is deferred.
        """
        if host in ("localhost", "localhost.localdomain"):
            return True
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return False  # not a literal IP; treat hostnames as allowed (see docstring)
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_reserved
            or ip.is_unspecified
        )

    # ------------------------------------------------------------------ crawl
    async def acrawl(self, request: CrawlRequest) -> CrawlResponse:
        """Crawl one URL on the service's event loop.

        This is the async core. The router's worker coroutines await it directly
        (they already run on this loop); the synchronous :meth:`crawl` bridges to
        it from another thread. The shared browser, semaphore, guards, timeout and
        recovery all live here so both entry points share one hardened path.
        """
        if self._is_closing:
            return CrawlResponse(url=request.url, error="service is closing", **_echo(request))
        reason = self._reject_reason(request)
        if reason is not None:
            return CrawlResponse(url=request.url, error=reason, **_echo(request))
        if self._pages_scraped >= self.config.max_pages_scraped:
            return CrawlResponse(
                url=request.url, error="service max_pages_scraped reached", **_echo(request)
            )

        async with self._semaphore:
            try:
                crawler = await self._ensure_crawler()
                run_cfg = self._run_config_factory(request)
                result = await asyncio.wait_for(
                    crawler.arun(url=request.url, config=run_cfg),
                    timeout=self.config.request_timeout,
                )
            except asyncio.TimeoutError:
                # Per-request timeout: do NOT tear down the shared browser.
                logger.warning("Crawl timed out for %s", request.url)
                return CrawlResponse(url=request.url, error="crawl timed out", **_echo(request))
            except Exception as exc:  # noqa: BLE001 - treated as browser-fatal
                logger.warning("Crawl failed (recovering) for %s: %s", request.url, exc)
                await self._recover()
                return CrawlResponse(url=request.url, error=str(exc), **_echo(request))

            self._pages_scraped += 1
            markdown = getattr(getattr(result, "markdown", None), "fit_markdown", "") or ""
            return CrawlResponse(
                url=request.url,
                markdown=markdown,
                # crawl4ai keeps the requested URL on ``result.url`` and the resolved,
                # post-redirect URL on ``result.redirected_url``. Provenance must record
                # where the content actually came from (the immutable version URL), so
                # prefer the redirected one.
                final_url=getattr(result, "redirected_url", None)
                or getattr(result, "url", None)
                or request.url,
                status=getattr(result, "status_code", None),
                links=_extract_links(result),
                **_echo(request),
            )

    def run_coroutine(self, coro) -> "Future":
        """Schedule ``coro`` on the service loop; return its concurrent Future.

        The bridge other components (the router) use to drive async work on the
        singleton browser's loop without owning a second loop/thread. Raises
        ``RuntimeError`` if the loop is not running (closing/closed).
        """
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def crawl(self, request: CrawlRequest) -> CrawlResponse:
        """Synchronous entry point. Never blocks the caller indefinitely."""
        if self._closed:
            return CrawlResponse(url=request.url, error="service is closed", **_echo(request))
        # Bridge budget covers worst case: init + request, plus headroom.
        budget = self.config.browser_init_timeout + self.config.request_timeout + 5.0
        try:
            future = self.run_coroutine(self.acrawl(request))
        except RuntimeError as exc:  # loop not running (closing/closed)
            return CrawlResponse(url=request.url, error=f"service unavailable: {exc}", **_echo(request))
        try:
            return future.result(timeout=budget)
        except FutureTimeoutError:
            future.cancel()
            return CrawlResponse(
                url=request.url, error="request timed out at the bridge", **_echo(request)
            )
        except Exception as exc:  # noqa: BLE001 - surface as an error response
            return CrawlResponse(url=request.url, error=f"bridge error: {exc}", **_echo(request))

    # ------------------------------------------------------------------ shutdown
    def close(self) -> None:
        """Graceful, idempotent shutdown of the browser and the loop."""
        if self._closed:
            return
        self._closed = True
        self._is_closing = True

        async def _shutdown() -> None:
            if self._crawler is not None:
                try:
                    await self._crawler.close()
                except Exception as exc:  # noqa: BLE001 - best-effort
                    logger.warning("Error closing crawler on shutdown: %s", exc)
                self._crawler = None

        try:
            fut = asyncio.run_coroutine_threadsafe(_shutdown(), self._loop)
            fut.result(timeout=self.config.browser_init_timeout)
        except (RuntimeError, FutureTimeoutError) as exc:
            logger.warning("Crawler shutdown incomplete: %s", exc)
        finally:
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=10.0)
            logger.info("CrawlService shut down.")
