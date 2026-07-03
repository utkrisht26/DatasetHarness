"""Remote fetcher — consume the crawl microservice through the ContentFetcher port.

Drop-in replacement for :class:`Crawl4AIContentFetcher`: the harness wires
``WebChannel(fetcher=RemoteContentFetcher(base_url=...))`` and everything downstream
(offload, provenance, saturation accounting) is unchanged, while the actual browser
lives in the service process.

Intelligence placement: relevance judging, saturation stop, and budgets run
*server-side* (they are the service's job). The ``on_result`` hook is still invoked
here for every returned page — post-hoc — so Engine A's coverage governor sees the
content and can skip later same-aspect sources; it just cannot flush this job
mid-crawl (the service's own saturation tracker covers that).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from .channel import FetchedPage
from .internal.contracts import CrawlRequest, CrawlResponse
from .internal.router import OnResult

_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


@dataclass
class RemoteContentFetcher:
    """Synchronous HTTP client for the crawl service (submit -> poll -> pages)."""

    base_url: str = "http://127.0.0.1:8000"
    timeout: float = 600.0  # end-to-end deadline per job
    poll_interval: float = 0.5
    judge: str = "bm25"  # server-side content judge: none | bm25 | llm
    llm_model: str | None = None  # used when judge == "llm"
    content_threshold: float = 0.35  # calibrated for the bm25 coverage judge
    max_depth: int = 2
    saturation: bool = True
    collapse_versions: bool = False
    pin_locales: bool = False
    # Injectable transport: any httpx.Client-compatible object (tests pass the
    # FastAPI TestClient). Built lazily against base_url when not supplied.
    client: Any = None
    _sleep: Any = field(default=time.sleep, repr=False)

    def fetch_source(
        self, request: CrawlRequest, *, on_result: OnResult | None = None, **_: Any
    ) -> list[FetchedPage]:
        payload = {
            "url": request.url,
            "max_pages": request.max_pages,
            "max_depth": self.max_depth,
            "keywords": list(request.keywords),
            "include_domains": list(request.include_domains or ()),
            "exclude_domains": list(request.exclude_domains or ()),
            "allowed_locales": list(request.allowed_locales),
            "respect_robots_txt": request.respect_robots_txt,
            "judge": self.judge,
            "content_threshold": self.content_threshold,
            "saturation": self.saturation,
            "collapse_versions": self.collapse_versions,
            "pin_locales": self.pin_locales,
        }
        if self.llm_model is not None:
            payload["llm_model"] = self.llm_model

        client = self._ensure_client()
        submitted = client.post("/api/jobs", json=payload)
        submitted.raise_for_status()
        job_id = submitted.json()["job_id"]

        job = self._poll(client, job_id)
        if job["status"] == "failed":
            raise RuntimeError(f"crawl job {job_id} failed: {job.get('error')}")

        pages = [
            FetchedPage(
                url=p["url"],
                markdown=p["markdown"],
                license_hint=p.get("license_hint"),
                relevance_score=p.get("relevance_score"),
                depth=p.get("depth", 0),
                fetched_at=p.get("fetched_at"),
                token_estimate=p.get("token_estimate"),
            )
            for p in job["pages"]
        ]
        if on_result is not None:
            # Post-hoc: feed the caller's governor so coverage accounting stays
            # correct; the crawl itself already finished server-side.
            for page in pages:
                on_result(
                    CrawlResponse(
                        url=page.url,
                        markdown=page.markdown,
                        license_hint=page.license_hint,
                        relevance_score=page.relevance_score,
                        depth=page.depth,
                        fetched_at=page.fetched_at,
                    )
                )
        return pages

    def fetch(self, url: str, *, max_pages: int = 1) -> list[FetchedPage]:
        return self.fetch_source(CrawlRequest(url=url, max_pages=max_pages))

    # ------------------------------------------------------------------ helpers
    def _poll(self, client: Any, job_id: str) -> dict:
        deadline = time.monotonic() + self.timeout
        while True:
            response = client.get(f"/api/jobs/{job_id}")
            response.raise_for_status()
            job = response.json()
            if job["status"] in _TERMINAL:
                return job
            if time.monotonic() >= deadline:
                client.post(f"/api/jobs/{job_id}/cancel")
                raise TimeoutError(f"crawl job {job_id} exceeded {self.timeout}s")
            self._sleep(self.poll_interval)

    def _ensure_client(self) -> Any:
        if self.client is None:
            import httpx

            self.client = httpx.Client(base_url=self.base_url, timeout=30.0)
        return self.client

    def close(self) -> None:
        if self.client is not None:
            self.client.close()
            self.client = None
