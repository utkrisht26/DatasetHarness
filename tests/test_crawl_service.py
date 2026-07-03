"""T2 acceptance tests for CrawlService — singleton, lock, timeout, recovery,
semaphore, SSRF/domain guards, close(). All run offline via an injected fake
crawler (no crawl4ai install required)."""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest

from adh.channels.web.internal.contracts import CrawlRequest
from adh.channels.web.internal.service import CrawlService, ServiceConfig


class _FakeMarkdown:
    def __init__(self, md: str):
        self.fit_markdown = md


class _FakeResult:
    def __init__(self, url: str, md: str):
        self.url = url
        self.markdown = _FakeMarkdown(md)
        self.status_code = 200


class FakeCrawler:
    """Async test double for AsyncWebCrawler with tunable behavior."""

    def __init__(self, *, arun_delay: float = 0.0, fail_times: int = 0, raise_exc=None):
        self.started = 0
        self.closed = 0
        self.calls = 0
        self.arun_delay = arun_delay
        self.fail_times = fail_times
        self.raise_exc = raise_exc
        self.max_in_flight = 0
        self._in_flight = 0

    async def start(self):
        self.started += 1

    async def close(self):
        self.closed += 1

    async def arun(self, url: str, config=None):
        self.calls += 1
        self._in_flight += 1
        self.max_in_flight = max(self.max_in_flight, self._in_flight)
        try:
            if self.fail_times > 0:
                self.fail_times -= 1
                raise (self.raise_exc or ConnectionError("browser died"))
            if self.arun_delay:
                await asyncio.sleep(self.arun_delay)
            return _FakeResult(url, f"# content for {url}")
        finally:
            self._in_flight -= 1


# run_config not needed for the fake; return None to avoid importing crawl4ai
def _no_run_config(_req):
    return None


def _service(factory, **cfg) -> CrawlService:
    return CrawlService(
        ServiceConfig(**cfg),
        crawler_factory=factory,
        run_config_factory=_no_run_config,
    )


def _req(url="https://example.com/p", **kw) -> CrawlRequest:
    return CrawlRequest(url=url, **kw)


def test_module_imports_without_crawl4ai():
    # The whole point: importable offline (the import above already proves it).
    import adh.channels.web.internal.service as svc

    assert hasattr(svc, "CrawlService")


def test_singleton_reused_across_calls():
    fake = FakeCrawler()
    svc = _service(lambda: fake)
    try:
        r1 = svc.crawl(_req("https://example.com/1"))
        r2 = svc.crawl(_req("https://example.com/2"))
        assert r1.error is None and r2.error is None
        assert fake.started == 1  # initialized once
        assert fake.calls == 2
        assert r1.markdown.startswith("# content")
    finally:
        svc.close()


def test_concurrent_cold_start_initializes_once():
    created = []

    def factory():
        c = FakeCrawler(arun_delay=0.05)
        created.append(c)
        return c

    svc = _service(factory, max_concurrent_pages=8)
    try:
        with ThreadPoolExecutor(max_workers=8) as ex:
            results = list(ex.map(lambda i: svc.crawl(_req(f"https://example.com/{i}")), range(8)))
        assert all(r.error is None for r in results)
        # Exactly one crawler created and started despite concurrent cold-start.
        assert len(created) == 1
        assert created[0].started == 1
    finally:
        svc.close()


def test_timeout_returns_error_without_tearing_down_browser():
    fake = FakeCrawler(arun_delay=5.0)
    svc = _service(lambda: fake, request_timeout=0.2)
    try:
        r = svc.crawl(_req())
        assert r.error and "timed out" in r.error
        # browser preserved: a subsequent (fast) call reuses the same instance
        fake.arun_delay = 0.0
        r2 = svc.crawl(_req())
        assert r2.error is None
        assert fake.started == 1  # NOT re-initialized
    finally:
        svc.close()


def test_fatal_error_triggers_recovery_and_closes_old():
    fake = FakeCrawler(fail_times=1, raise_exc=ConnectionError("browser crashed"))
    created = [fake]

    def factory():
        if created:
            return created.pop(0)
        nxt = FakeCrawler()
        created_after.append(nxt)
        return nxt

    created_after: list[FakeCrawler] = []
    svc = _service(factory)
    try:
        r1 = svc.crawl(_req())  # fatal error -> recover
        assert r1.error and "crashed" in r1.error
        assert fake.closed == 1  # old crawler closed on recovery (no leak)
        r2 = svc.crawl(_req())  # re-init with a fresh crawler
        assert r2.error is None
        assert created_after and created_after[0].started == 1
    finally:
        svc.close()


def test_semaphore_caps_concurrency():
    fake = FakeCrawler(arun_delay=0.1)
    svc = _service(lambda: fake, max_concurrent_pages=2)
    try:
        with ThreadPoolExecutor(max_workers=6) as ex:
            list(ex.map(lambda i: svc.crawl(_req(f"https://example.com/{i}")), range(6)))
        assert fake.max_in_flight <= 2
    finally:
        svc.close()


@pytest.mark.parametrize(
    "url,kwargs,needle",
    [
        ("file:///etc/passwd", {}, "scheme"),
        ("https://reddit.com/x", {"exclude_domains": ("reddit.com",)}, "exclude_domains"),
        ("https://evil.com/x", {"include_domains": ("docs.rs",)}, "include_domains"),
        ("http://localhost/x", {}, "private/internal"),
        ("http://127.0.0.1/x", {}, "private/internal"),
        ("http://169.254.169.254/latest/meta-data", {}, "private/internal"),
    ],
)
def test_ssrf_and_domain_guards(url, kwargs, needle):
    fake = FakeCrawler()
    svc = _service(lambda: fake)
    try:
        r = svc.crawl(_req(url, **kwargs))
        assert r.error and needle in r.error
        assert fake.calls == 0  # never fetched
    finally:
        svc.close()


def test_max_pages_safety_cap():
    fake = FakeCrawler()
    svc = _service(lambda: fake, max_pages_scraped=1)
    try:
        r1 = svc.crawl(_req("https://example.com/1"))
        r2 = svc.crawl(_req("https://example.com/2"))
        assert r1.error is None
        assert r2.error and "max_pages_scraped" in r2.error
    finally:
        svc.close()


def test_close_is_idempotent():
    fake = FakeCrawler()
    svc = _service(lambda: fake)
    svc.crawl(_req())
    svc.close()
    svc.close()  # second close must be a no-op, not raise
    assert fake.closed == 1
    # calls after close return an error, not a hang
    r = svc.crawl(_req())
    assert r.error and "closed" in r.error
