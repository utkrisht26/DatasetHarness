"""Crawl router — today's crawl-service endpoints as a mountable APIRouter (T3).

The endpoint logic moved here verbatim from ``adh.channels.web.app``; the engine
(:class:`CrawlServiceApp`) and its request/response models stay there. Paths are
router-relative so the mount point owns the prefix:

* gateway (``adh.service.app``): ``/api/crawl/jobs`` … (decision 5's contract)
* legacy standalone app (``adh.channels.web.app.create_app``): ``/api/jobs`` …
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from fastapi import APIRouter, HTTPException

from adh.channels.web.app import CrawlJobRequest, JobOut, JobSummaryOut, PageOut

if TYPE_CHECKING:  # pragma: no cover - typing only (avoids a runtime import cycle)
    from adh.channels.web.app import CrawlServiceApp


def build_crawl_router(service: "CrawlServiceApp") -> APIRouter:
    """All crawl endpoints over one shared :class:`CrawlServiceApp`."""
    router = APIRouter(tags=["crawl"])

    @router.post("/jobs", status_code=202)
    def submit_job(request: CrawlJobRequest) -> dict:
        return {"job_id": service.submit(request)}

    @router.get("/jobs")
    def list_jobs(status: str | None = None) -> list[JobSummaryOut]:
        with service._lock:
            records = list(service._jobs.items())

        summaries = []
        for job_id, record in records:
            with record._lock:
                if status and record.status != status:
                    continue
                summaries.append(
                    JobSummaryOut(
                        job_id=job_id,
                        status=record.status,  # type: ignore[arg-type]
                        url=record.request.url,
                        n_pages=len(record.pages),
                        pages_fetched=record.pages_fetched,
                        duration=_duration(record),
                        saturated=record.saturated,
                        created_at=record.created_at,
                    )
                )
        summaries.sort(key=lambda s: s.created_at, reverse=True)
        return summaries

    @router.get("/jobs/{job_id}")
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
                duration=_duration(record),
                pages_fetched=record.pages_fetched,
            )

    @router.post("/jobs/{job_id}/cancel")
    def cancel_job(job_id: str) -> dict:
        if service.get(job_id) is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        return {"cancelled": service.cancel(job_id)}

    @router.get("/jobs/{job_id}/events")
    def get_job_events(job_id: str, since: int = 0) -> dict:
        record = service.get(job_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"unknown job {job_id!r}")
        with record._lock:
            events = record.events[since:]
            next_since = len(record.events)
        return {"events": events, "next_since": next_since}

    @router.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok", "jobs": service.counts()}

    @router.get("/usage")
    def usage() -> dict:
        return {"llm": service.usage.to_dict(), "jobs": service.counts()}

    return router


def _duration(record) -> float:
    if record.started_at is None:
        return 0.0
    end = record.completed_at if record.completed_at is not None else time.time()
    return end - record.started_at
