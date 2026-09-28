"""The gateway app factory — one thin deployable, per-domain routers (T3).

Locked decision 5: the frontend talks to one origin; each domain lives under its
own prefix (``/api/crawl/*``, ``/api/runs/*``) so lifting one to its own process
later is a config flip, not an API change.

The factory is thin: CORS, lifespan teardown, router includes, SPA mount. All
collaborators are injectable for offline tests; the defaults wire the live stack
(deepagents discovery with heuristic fallback, live channels when extras exist).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from adh.channels.web.app import CrawlServiceApp, JudgeFactory
from adh.runtime import (
    live_agent_factory as _live_agent_factory,
    live_channels as _live_channels,
    live_ports as _live_ports,  # noqa: F401  # re-exported for callers/tests that patch it
)
from adh.settings import get_settings

from .routers.crawl import build_crawl_router
from .routers.runs import RunsService, build_runs_router
from .routers.specs import build_specs_router
from .security import BearerTokenMiddleware
from .spa import mount_spa

_CONFIGURED_DISCOVERY_MODEL = object()


def create_app(
    *,
    crawl_service: CrawlServiceApp | None = None,
    runs_service: RunsService | None = None,
    runs_dir: str | Path = "./runs",
    discovery_model: str | None | object = _CONFIGURED_DISCOVERY_MODEL,
    fetcher=None,
    llm_judge_factory: JudgeFactory | None = None,
    workers: int = 2,
    api_key: str | None = None,
) -> FastAPI:
    """Build the gateway app.

    The default discovery model comes from :mod:`adh.settings`, while an
    explicit ``discovery_model=None`` selects heuristic-only discovery. A
    non-empty ``api_key`` (or ``ADH_API_KEY``) protects every API route except
    the top-level liveness probe.
    """
    settings = get_settings()
    configured_discovery_model = (
        settings.discovery_model
        if discovery_model is _CONFIGURED_DISCOVERY_MODEL
        else cast(str | None, discovery_model)
    )
    crawl = crawl_service or CrawlServiceApp(
        fetcher, llm_judge_factory=llm_judge_factory, workers=workers
    )
    runs = runs_service or RunsService(
        runs_dir,
        agent_factory=_live_agent_factory(configured_discovery_model),
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

    configured_api_key = api_key if api_key is not None else settings.service_api_key_value
    app.state.auth_enabled = bool(configured_api_key)
    if configured_api_key:
        # Add auth before CORS so CORS remains the outer middleware and browser
        # clients receive the appropriate headers even on a 401 response.
        app.add_middleware(BearerTokenMiddleware, token=configured_api_key)

    # CORS is opt-in via ADH_CORS_ORIGINS (comma-separated). The bundled SPA is
    # same-origin and needs none; wildcard origins with credentials are never
    # combined (browsers reject the pair, and it would be a CSRF hazard anyway).
    cors_origins = list(settings.cors_origins)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "Authorization"],
    )

    app.include_router(build_crawl_router(crawl), prefix="/api/crawl")
    app.include_router(build_runs_router(runs), prefix="/api/runs")
    app.include_router(build_specs_router(), prefix="/api/specs")

    @app.get("/api/healthz")
    def healthz() -> dict:
        return {
            "status": "ok",
            "auth_enabled": app.state.auth_enabled,
            "crawl_jobs": crawl.counts(),
        }

    mount_spa(app)
    return app
