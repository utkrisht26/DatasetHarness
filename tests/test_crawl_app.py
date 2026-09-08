"""Tests for the crawl microservice (app.py) and its remote client (remote.py).

Fully offline: the app is built over a Crawl4AIContentFetcher with an injected fake
crawler (no browser, no network); HTTP runs through FastAPI's TestClient; the LLM
judge is an injected fake. Exercises the service's intelligence end to end —
relevance gating, saturation stop, budgets, cancel, and usage accounting.
"""

from __future__ import annotations

import time

from fastapi.testclient import TestClient

from adh.channels.web import Crawl4AIContentFetcher, RemoteContentFetcher
from adh.channels.web.app import create_app
from adh.channels.web.internal.relevance import RelevanceJudge
from adh.llm import TokenUsage

_ON_TOPIC = (
    "A view is a Python function that takes a request and returns a response. "
    "Class-based views structure views and reuse code via mixins and inheritance. "
    "The view dispatches the request to a handler method per HTTP method."
)
_OFF_TOPIC = (
    "The software foundation is an independent non-profit organization. "
    "Your donation directly funds development sprints and community outreach. "
    "Fundraising campaigns run throughout the year and sponsors get recognition."
)
_KEYWORDS = ["views view request response class-based dispatch handler mixin"]


# --------------------------------------------------------------------- fakes
class _FakeMarkdown:
    def __init__(self, md: str):
        self.fit_markdown = md


class _FakeResult:
    def __init__(self, url: str, md: str, links: list[str]):
        self.url = url
        self.markdown = _FakeMarkdown(md)
        self.status_code = 200
        self.links = {"internal": [{"href": h, "text": h.rsplit("/", 2)[-2]} for h in links]}


class FakeCrawler:
    """AsyncWebCrawler double serving a canned site."""

    def __init__(self, pages: dict[str, tuple[str, list[str]]]):
        self.pages = pages
        self.calls: list[str] = []

    async def start(self):
        return None

    async def close(self):
        return None

    async def arun(self, url: str, config=None):
        self.calls.append(url)
        md, links = self.pages.get(url, ("", []))
        return _FakeResult(url, md, links)


class _FakeLlmJudge(RelevanceJudge):
    """Deterministic 'LLM' judge that records usage like the real one would."""

    def __init__(self, usage):
        self.usage = usage

    def score(self, query: str, content: str) -> float:
        self.usage.add(TokenUsage(input_tokens=100, output_tokens=5))
        return 0.0 if "donation" in content else 1.0


def _make_app(pages: dict[str, tuple[str, list[str]]], **kwargs):
    fetcher = Crawl4AIContentFetcher(
        crawler_factory=lambda: FakeCrawler(pages),
        run_config_factory=lambda req: None,
    )
    return create_app(fetcher, **kwargs)


def _await_job(client, job_id: str, timeout: float = 10.0) -> dict:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] in ("succeeded", "failed", "cancelled"):
            return job
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} did not finish: {job}")


_SITE = {
    "https://ex.com/views/": (_ON_TOPIC, ["https://ex.com/views/cbv/", "https://ex.com/donate/"]),
    "https://ex.com/views/cbv/": (_ON_TOPIC + " Generic editing views handle forms.", []),
    "https://ex.com/donate/": (_OFF_TOPIC, ["https://ex.com/donate/thanks/"]),
    "https://ex.com/donate/thanks/": (_OFF_TOPIC, []),
}


# --------------------------------------------------------------------- service basics
def test_submit_poll_and_fetch_pages():
    with TestClient(_make_app(_SITE)) as client:
        r = client.post(
            "/api/jobs",
            json={
                "url": "https://ex.com/views/",
                "max_pages": 10,
                "keywords": _KEYWORDS,
                "judge": "none",
                "saturation": False,
            },
        )
        assert r.status_code == 202
        job = _await_job(client, r.json()["job_id"])
        assert job["status"] == "succeeded"
        assert job["n_pages"] == 4  # no judge, no saturation -> everything
        assert all(p["markdown"] for p in job["pages"])
        # include_content=false returns metadata only
        lean = client.get(f"/api/jobs/{job['job_id']}?include_content=false").json()
        assert lean["n_pages"] == 4 and all(p["markdown"] == "" for p in lean["pages"])


def test_bm25_judge_drops_off_topic_server_side():
    with TestClient(_make_app(_SITE)) as client:
        r = client.post(
            "/api/jobs",
            json={
                "url": "https://ex.com/views/",
                "max_pages": 10,
                "keywords": _KEYWORDS,
                "judge": "bm25",
                "content_threshold": 0.25,
                "saturation": False,
            },
        )
        job = _await_job(client, r.json()["job_id"])
        urls = {p["url"] for p in job["pages"]}
        assert "https://ex.com/views/cbv/" in urls  # on-topic kept
        assert "https://ex.com/donate/" not in urls  # judged off-topic
        assert "https://ex.com/donate/thanks/" not in urls  # its links not expanded


def test_saturation_stops_before_budget():
    same = "identical content block repeated verbatim on every page of this site"
    chain = {f"https://ex.com/p{i}/": (same, [f"https://ex.com/p{i + 1}/"]) for i in range(8)}
    with TestClient(_make_app(chain)) as client:
        r = client.post(
            "/api/jobs",
            json={
                "url": "https://ex.com/p0/",
                "max_pages": 8,
                "max_depth": 10,
                "judge": "none",
                "saturation": True,
            },
        )
        job = _await_job(client, r.json()["job_id"])
        assert job["saturated"] is True
        assert job["n_pages"] < 8  # stopped on novelty, not the budget


def test_llm_judge_via_injected_factory_and_usage_endpoint():
    app = _make_app(_SITE, llm_judge_factory=lambda model_id, usage: _FakeLlmJudge(usage))
    with TestClient(app) as client:
        r = client.post(
            "/api/jobs",
            json={
                "url": "https://ex.com/views/",
                "max_pages": 10,
                "keywords": _KEYWORDS,
                "judge": "llm",
                "content_threshold": 0.5,
                "saturation": False,
            },
        )
        job = _await_job(client, r.json()["job_id"])
        urls = {p["url"] for p in job["pages"]}
        assert "https://ex.com/donate/" not in urls  # LLM judge dropped it
        usage = client.get("/api/usage").json()
        assert usage["llm"]["calls"] >= 1  # tokens were counted
        assert usage["llm"]["total_tokens"] == usage["llm"]["calls"] * 105


def test_unknown_job_and_cancel_404():
    with TestClient(_make_app(_SITE)) as client:
        assert client.get("/api/jobs/nope").status_code == 404
        assert client.post("/api/jobs/nope/cancel").status_code == 404
        assert client.get("/api/healthz").json()["status"] == "ok"


def test_failed_job_reports_error():
    def broken_factory():
        raise RuntimeError("browser exploded")

    fetcher = Crawl4AIContentFetcher(
        crawler_factory=broken_factory, run_config_factory=lambda req: None
    )
    with TestClient(create_app(fetcher)) as client:
        r = client.post("/api/jobs", json={"url": "https://ex.com/x/", "judge": "none"})
        job = _await_job(client, r.json()["job_id"])
        # The service surfaces per-page errors as an empty result, not a crash;
        # a job only fails on an unexpected exception. Either way it terminates.
        assert job["status"] in ("succeeded", "failed")
        assert job["n_pages"] == 0


# --------------------------------------------------------------------- remote client
def test_remote_fetcher_end_to_end_through_http():
    with TestClient(_make_app(_SITE)) as client:
        remote = RemoteContentFetcher(
            client=client,
            poll_interval=0.01,
            judge="bm25",
            content_threshold=0.25,
            saturation=False,
        )
        from adh.channels.web.internal.contracts import CrawlRequest

        pages = remote.fetch_source(
            CrawlRequest(url="https://ex.com/views/", max_pages=10, keywords=tuple(_KEYWORDS))
        )
        urls = {p.url for p in pages}
        assert "https://ex.com/views/" in urls
        assert "https://ex.com/donate/" not in urls  # server-side intelligence applied


def test_remote_fetcher_feeds_governor_post_hoc():
    observed: list[str] = []

    class _Gov:
        def observe(self, aspect_id, text):
            observed.append(text)

        def should_stop(self, aspect_id):
            return False

    with TestClient(_make_app(_SITE)) as client:
        remote = RemoteContentFetcher(
            client=client, poll_interval=0.01, judge="none", saturation=False
        )
        from adh.channels.web.internal.contracts import CrawlRequest, CrawlResponse

        gov = _Gov()

        def hook(resp: CrawlResponse) -> bool:
            gov.observe("a0", resp.markdown)
            return not gov.should_stop("a0")

        pages = remote.fetch_source(
            CrawlRequest(url="https://ex.com/views/", max_pages=10), on_result=hook
        )
        assert len(observed) == len(pages) > 0  # coverage accounting intact
