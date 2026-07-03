"""The crawl microservice — the web channel as a deployable HTTP service (OUTLINE §1.1).

This is the design doc's "Hands" behind a real process boundary. A caller states
*what* it wants (seed URL, topic keywords, budgets); the service owns *how*:

* **relevance** — per-job content judge (``bm25`` default, ``llm`` optional, ``none``)
  dropping off-topic pages and their links;
* **saturation** — the coverage-saturation stop (design-doc §2 loop): the job halts
  when new pages stop adding novel information, not when a page count runs out;
* **budgets** — ``max_pages`` (kept) and ``max_fetches`` (fetched) caps;
* **accounting** — LLM token usage aggregated across jobs at ``GET /api/usage``.

Endpoints: ``POST /api/jobs`` (submit, 202 + id) · ``GET /api/jobs`` (list) · ``GET /api/jobs/{id}`` (status/results) ·
``POST /api/jobs/{id}/cancel`` (flush the frontier) · ``GET /api/jobs/{id}/events`` · ``GET /api/healthz`` · ``GET /api/usage``.

One browser (one :class:`CrawlService`) serves all jobs; job execution runs on a small
thread pool over the synchronous bridge. Everything is injectable for offline tests.
"""

from __future__ import annotations

import logging
import os
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Callable, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from adh.llm import DEFAULT_CHAT_MODEL, UsageTracker
from adh.orchestration.saturation import SaturationTracker

from .channel import Crawl4AIContentFetcher, FetchedPage
from .internal.contracts import CrawlRequest, CrawlResponse
from .internal.policy import assemble_policy
from .internal.relevance import Bm25RelevanceJudge, KeepAllJudge, LlmRelevanceJudge, RelevanceJudge

logger = logging.getLogger(__name__)

# A judge factory turns a model id into a RelevanceJudge (injectable in tests).
JudgeFactory = Callable[[str, UsageTracker], RelevanceJudge]

# Fetch-budget headroom when the caller doesn't set max_fetches explicitly: allow
# the judge to discard up to 3 pages for every kept one before the job is cut off.
_FETCH_HEADROOM = 4


class CrawlJobRequest(BaseModel):
    """What a client asks the service for. Everything beyond ``url`` is steering."""

    model_config = ConfigDict(extra="forbid")

    url: str
    max_pages: int = Field(8, ge=1, le=500)
    max_fetches: int | None = Field(None, ge=1)  # default: _FETCH_HEADROOM * max_pages
    max_depth: int = Field(2, ge=0, le=10)  # how far outlinks may travel from the seed
    keywords: list[str] = Field(default_factory=list)  # topic vocabulary for relevance
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    allowed_locales: list[str] = Field(default_factory=list)
    respect_robots_txt: bool = True
    # Relevance: which judge gates fetched content, and how strictly.
    judge: Literal["none", "bm25", "llm"] = "bm25"
    llm_model: str = DEFAULT_CHAT_MODEL  # used only when judge == "llm"
    content_threshold: float = Field(0.35, ge=0.0, le=1.0)  # calibrated for bm25
    # Saturation-driven stop (novelty tracking) — the primary stop signal.
    saturation: bool = True
    # Opt-in site-shape strategies (see internal.policy).
    collapse_versions: bool = False
    pin_locales: bool = False


class PageOut(BaseModel):
    url: str
    markdown: str
    license_hint: str | None = None
    relevance_score: float | None = None
    depth: int = 0
    fetched_at: float | None = None
    token_estimate: int | None = None


class JobOut(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    url: str
    pages: list[PageOut] = Field(default_factory=list)
    n_pages: int = 0
    saturated: bool = False
    error: str | None = None
    created_at: float | None = None
    started_at: float | None = None
    completed_at: float | None = None
    duration: float = 0.0
    pages_fetched: int = 0


class JobSummaryOut(BaseModel):
    job_id: str
    status: Literal["queued", "running", "succeeded", "failed", "cancelled"]
    url: str
    n_pages: int
    pages_fetched: int
    duration: float
    saturated: bool
    created_at: float


@dataclass
class _JobRecord:
    """Server-side state for one submitted crawl job."""

    request: CrawlJobRequest
    status: str = "queued"
    pages: list[FetchedPage] = field(default_factory=list)
    saturated: bool = False
    error: str | None = None
    cancelled: bool = field(default=False)
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    pages_fetched: int = 0
    events: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self._lock = threading.Lock()


def _default_llm_judge(model_id: str, usage: UsageTracker) -> RelevanceJudge:
    return LlmRelevanceJudge.from_model_id(model_id, usage=usage)


class CrawlServiceApp:
    """Owns the shared fetcher, the job registry, and the worker pool."""

    def __init__(
        self,
        fetcher: Crawl4AIContentFetcher | None = None,
        *,
        llm_judge_factory: JudgeFactory | None = None,
        workers: int = 2,
    ) -> None:
        self.fetcher = fetcher or Crawl4AIContentFetcher()
        self.usage = UsageTracker()
        self._llm_judge_factory = llm_judge_factory or _default_llm_judge
        self._executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="crawl-job")
        self._jobs: dict[str, _JobRecord] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ job API
    def submit(self, request: CrawlJobRequest) -> str:
        job_id = f"crawl-{uuid.uuid4().hex[:12]}"
        with self._lock:
            self._jobs[job_id] = _JobRecord(request=request)
        self._executor.submit(self._run, job_id)
        return job_id

    def get(self, job_id: str) -> _JobRecord | None:
        with self._lock:
            return self._jobs.get(job_id)

    def cancel(self, job_id: str) -> bool:
        record = self.get(job_id)
        if record is None:
            return False
        with record._lock:
            if record.status in ("succeeded", "failed", "cancelled"):
                return False
            record.cancelled = True
        # The seed carries job_id as its aspect_id, so this flushes exactly this job.
        self.fetcher.cancel_aspect(job_id)
        return True

    def counts(self) -> dict:
        with self._lock:
            out: dict[str, int] = {}
            for record in self._jobs.values():
                with record._lock:
                    out[record.status] = out.get(record.status, 0) + 1
            return out

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        self.fetcher.close()

    # ------------------------------------------------------------------ execution
    def _run(self, job_id: str) -> None:
        record = self.get(job_id)
        if record is None:
            return
        
        with record._lock:
            if record.cancelled:  # cancelled while still queued
                record.status = "cancelled"
                record.completed_at = time.time()
                return
            record.status = "running"
            record.started_at = time.time()
            record.events.append({
                "timestamp": time.time(),
                "url": record.request.url,
                "verdict": "running",
                "metadata": {"depth": 0}
            })
            record.pages_fetched = 1

        req = record.request
        try:
            tracker = SaturationTracker() if req.saturation else None

            def on_result(resp: CrawlResponse) -> bool:
                with record._lock:
                    if record.cancelled:
                        return False
                if tracker is None:
                    return True
                tracker.update(resp.markdown)
                if tracker.is_saturated():
                    with record._lock:
                        record.saturated = True
                        record.events.append({
                            "timestamp": time.time(),
                            "url": resp.final_url or resp.url,
                            "verdict": "saturated",
                            "metadata": {"reason": "Novelty threshold reached (coverage saturated)", "depth": resp.depth}
                        })
                    return False
                return True

            def on_event(verdict: str, url: str, metadata: dict):
                with record._lock:
                    if verdict == "running":
                        record.pages_fetched += 1
                    record.events.append({
                        "timestamp": time.time(),
                        "url": url,
                        "verdict": verdict,
                        "metadata": metadata
                    })

            seed = CrawlRequest(
                url=req.url,
                max_pages=req.max_pages,
                aspect_id=job_id,  # cancel() flushes by this id
                respect_robots_txt=req.respect_robots_txt,
                include_domains=tuple(req.include_domains) or None,
                exclude_domains=tuple(req.exclude_domains) or None,
                allowed_locales=tuple(req.allowed_locales),
                keywords=tuple(req.keywords),
            )
            policy = assemble_policy(
                seed,
                judge=self._make_judge(req),
                content_threshold=req.content_threshold if req.judge != "none" else 0.0,
                collapse_versions=req.collapse_versions,
                pin_locales=req.pin_locales,
            )
            pages = self.fetcher.fetch_source(
                seed,
                on_result=on_result,
                policy=policy,
                max_fetches=req.max_fetches or _FETCH_HEADROOM * req.max_pages,
                max_depth=req.max_depth,
                on_event=on_event,
            )
            with record._lock:
                record.pages = pages
                record.status = "cancelled" if record.cancelled else "succeeded"
        except Exception as exc:  # noqa: BLE001 - job failure must not kill the worker
            logger.exception("crawl job %s failed", job_id)
            with record._lock:
                record.error = str(exc)
                record.status = "failed"
                record.events.append({
                    "timestamp": time.time(),
                    "url": req.url,
                    "verdict": "failed",
                    "metadata": {"error": str(exc), "reason": "Job execution failed"}
                })
        finally:
            with record._lock:
                record.completed_at = time.time()

    def _make_judge(self, req: CrawlJobRequest) -> RelevanceJudge:
        if req.judge == "bm25":
            return Bm25RelevanceJudge()
        if req.judge == "llm":
            return self._llm_judge_factory(req.llm_model, self.usage)
        return KeepAllJudge()


def create_app(
    fetcher: Crawl4AIContentFetcher | None = None,
    *,
    llm_judge_factory: JudgeFactory | None = None,
    workers: int = 2,
) -> FastAPI:
    """Build the FastAPI app. All collaborators are injectable for offline tests."""
    service = CrawlServiceApp(fetcher, llm_judge_factory=llm_judge_factory, workers=workers)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        service.close()  # graceful browser + pool teardown on shutdown

    app = FastAPI(title="adh crawl service", lifespan=lifespan)
    app.state.service = service

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/api/jobs", status_code=202)
    def submit_job(request: CrawlJobRequest) -> dict:
        return {"job_id": service.submit(request)}

    @app.get("/api/jobs")
    def list_jobs(status: str | None = None) -> list[JobSummaryOut]:
        with service._lock:
            records = list(service._jobs.items())
            
        summaries = []
        for job_id, record in records:
            with record._lock:
                if status and record.status != status:
                    continue
                dur = 0.0
                if record.started_at is not None:
                    if record.completed_at is not None:
                        dur = record.completed_at - record.started_at
                    else:
                        dur = time.time() - record.started_at
                summaries.append(
                    JobSummaryOut(
                        job_id=job_id,
                        status=record.status,  # type: ignore[arg-type]
                        url=record.request.url,
                        n_pages=len(record.pages),
                        pages_fetched=record.pages_fetched,
                        duration=dur,
                        saturated=record.saturated,
                        created_at=record.created_at,
                    )
                )
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    @app.get("/api/jobs/{job_id}")
    def get_job(job_id: str, include_content: bool = True) -> JobOut:
        record = service.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        with record._lock:
            pages = [
                PageOut(
                    url=p.url,
                    markdown=p.markdown if include_content else "",
                    license_hint=p.license_hint,
                    relevance_score=p.relevance_score,
                    depth=p.depth,
                    fetched_at=p.fetched_at,
                    token_estimate=p.token_estimate,
                )
                for p in record.pages
            ]
            dur = 0.0
            if record.started_at is not None:
                if record.completed_at is not None:
                    dur = record.completed_at - record.started_at
                else:
                    dur = time.time() - record.started_at
            return JobOut(
                job_id=job_id,
                status=record.status,  # type: ignore[arg-type]
                url=record.request.url,
                pages=pages,
                n_pages=len(record.pages),
                saturated=record.saturated,
                error=record.error,
                created_at=record.created_at,
                started_at=record.started_at,
                completed_at=record.completed_at,
                duration=dur,
                pages_fetched=record.pages_fetched,
            )

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        if service.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return {"cancelled": service.cancel(job_id)}

    @app.get("/api/jobs/{job_id}/events")
    def get_job_events(job_id: str, since: int = 0) -> dict:
        record = service.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        with record._lock:
            events = record.events[since:]
            next_since = len(record.events)
        return {"events": events, "next_since": next_since}

    @app.get("/api/healthz")
    def healthz() -> dict:
        return {"status": "ok", "jobs": service.counts()}

    @app.get("/api/usage")
    def usage() -> dict:
        return {"llm": service.usage.to_dict(), "jobs": service.counts()}

    # Static mount fallback routing for React Router SPA (mounted last)
    frontend_dist = os.path.join(os.getcwd(), "frontend/dist")
    if not os.path.exists(frontend_dist):
        sibling_path = os.path.join(os.path.dirname(os.getcwd()), "adh-frontend", "dist")
        if os.path.exists(sibling_path):
            frontend_dist = sibling_path

    if os.path.exists(frontend_dist):
        app.mount("/assets", StaticFiles(directory=os.path.join(frontend_dist, "assets")), name="assets")

        @app.get("/{catchall:path}", include_in_schema=False)
        def serve_spa(catchall: str):
            if catchall.startswith(("api/", "api", "docs", "openapi.json", "healthz", "usage")):
                raise HTTPException(status_code=404, detail="Not Found")
            file_path = os.path.join(frontend_dist, catchall)
            if catchall and os.path.exists(file_path) and os.path.isfile(file_path):
                return FileResponse(file_path)
            return FileResponse(os.path.join(frontend_dist, "index.html"))

    return app
