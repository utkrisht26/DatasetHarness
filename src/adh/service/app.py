"""The gateway app factory — one thin deployable, per-domain routers (T3).

Locked decision 5: the frontend talks to one origin; each domain lives under its
own prefix (``/api/crawl/*``, ``/api/runs/*``) so lifting one to its own process
later is a config flip, not an API change.

The factory is thin: CORS, lifespan teardown, router includes, SPA mount. All
collaborators are injectable for offline tests; the defaults wire the live stack
(deepagents discovery with heuristic fallback, live channels when extras exist).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adh.channels.web.app import CrawlServiceApp, JudgeFactory
from adh.llm import DEFAULT_CHAT_MODEL
from adh.runtime import (
    live_agent_factory as _live_agent_factory,
    live_channels as _live_channels,
    live_ports as _live_ports,  # noqa: F401  # re-exported for callers/tests that patch it
)

from .routers.crawl import build_crawl_router
from .routers.runs import RunsService, build_runs_router
from .spa import mount_spa


def create_app(
    *,
    crawl_service: CrawlServiceApp | None = None,
    runs_service: RunsService | None = None,
    runs_dir: str | Path = "./runs",
    discovery_model: str | None = DEFAULT_CHAT_MODEL,
    fetcher=None,
    llm_judge_factory: JudgeFactory | None = None,
    workers: int = 2,
) -> FastAPI:
    """Build the gateway app. ``discovery_model=None`` -> heuristic-only discovery."""
    crawl = crawl_service or CrawlServiceApp(
        fetcher, llm_judge_factory=llm_judge_factory, workers=workers
    )
    runs = runs_service or RunsService(
        runs_dir,
        agent_factory=_live_agent_factory(discovery_model),
        channels=_live_channels(),
        workers=workers,
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        yield
        crawl.close()
        runs.close()

    app = FastAPI(title="adh service", lifespan=lifespan)
    app.state.crawl = crawl
    app.state.runs = runs

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(build_crawl_router(crawl), prefix="/api/crawl")
    app.include_router(build_runs_router(runs), prefix="/api/runs")

    @app.get("/api/healthz")
    def healthz() -> dict:
        return {"status": "ok", "crawl_jobs": crawl.counts()}

    mount_spa(app)
    return app
