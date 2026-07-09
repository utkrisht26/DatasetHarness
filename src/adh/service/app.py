"""The gateway app factory — one thin deployable, per-domain routers (T3).

Locked decision 5: the frontend talks to one origin; each domain lives under its
own prefix (``/api/crawl/*``, ``/api/runs/*``) so lifting one to its own process
later is a config flip, not an API change.

The factory is thin: CORS, lifespan teardown, router includes, SPA mount. All
collaborators are injectable for offline tests; the defaults wire the live stack
(deepagents discovery with heuristic fallback, live channels when extras exist).
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adh.channels.web.app import CrawlServiceApp, JudgeFactory
from adh.llm import DEFAULT_CHAT_MODEL, UsageTracker

from .routers.crawl import build_crawl_router
from .routers.runs import RunsService, build_runs_router
from .spa import mount_spa

logger = logging.getLogger(__name__)


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


# ------------------------------------------------------------- live defaults
def _live_agent_factory(discovery_model: str | None):
    """Deep discovery over live ports when configured; heuristic otherwise.

    The factory receives the run's :class:`UsageTracker` so every deepagents
    model call is metered into that run's metrics (decision 4).
    """
    from adh.discovery import (
        DeepAgentDiscoveryAgent,
        FallbackDiscoveryAgent,
        SourceDiscoveryAgent,
    )

    search, registry = _live_ports()

    def factory(usage: UsageTracker):
        heuristic = SourceDiscoveryAgent(search=search, registry=registry)
        if discovery_model is None:
            return heuristic
        return FallbackDiscoveryAgent(
            primary=DeepAgentDiscoveryAgent(
                model=discovery_model, search=search, registry=registry, usage=usage
            ),
            fallback=heuristic,
        )

    return factory


def _live_ports():
    """Best live search/registry ports available; in-memory stubs otherwise."""
    from adh.search import InMemoryDatasetRegistry, InMemorySearchProvider

    search = InMemorySearchProvider()  # no live web meta-search adapter yet
    try:
        from adh.search.hf_registry import HfDatasetRegistry

        registry = HfDatasetRegistry()
    except Exception as exc:  # pragma: no cover - needs the datasets extra
        logger.warning("HF dataset registry unavailable (%s); using in-memory stub", exc)
        registry = InMemoryDatasetRegistry()
    return search, registry


def _live_channels() -> dict:
    """Wire real acquisition channels when their extras are importable."""
    from adh.contract import SourceChannel

    channels: dict = {}
    try:
        from adh.channels import WebChannel
        from adh.channels.web import Crawl4AIContentFetcher

        channels[SourceChannel.WEB] = WebChannel(fetcher=Crawl4AIContentFetcher())
    except Exception as exc:  # pragma: no cover - needs the web extra
        print(f"[adh] web channel unavailable: {exc}", file=sys.stderr)
    try:
        from adh.channels import DatasetsChannel, HfDatasetLoader

        channels[SourceChannel.DATASETS] = DatasetsChannel(loader=HfDatasetLoader())
    except Exception as exc:  # pragma: no cover - needs the datasets extra
        print(f"[adh] datasets channel unavailable: {exc}", file=sys.stderr)
    return channels
