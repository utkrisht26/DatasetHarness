"""Static SPA mount for the adh-frontend build (React Router catch-all).

Shared by the gateway app and the legacy standalone crawl app so the fallback
routing logic lives in exactly one place.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles


def find_frontend_dist() -> str | None:
    """Locate a built frontend: ./frontend/dist or the sibling adh-frontend repo."""
    local = os.path.join(os.getcwd(), "frontend/dist")
    if os.path.exists(local):
        return local
    sibling = os.path.join(os.path.dirname(os.getcwd()), "adh-frontend", "dist")
    if os.path.exists(sibling):
        return sibling
    return None


def mount_spa(app: FastAPI, dist_dir: str | None = None) -> None:
    """Serve the SPA with API-path passthrough. No-op when no build exists."""
    frontend_dist = dist_dir or find_frontend_dist()
    if frontend_dist is None:
        return

    app.mount(
        "/assets",
        StaticFiles(directory=os.path.join(frontend_dist, "assets")),
        name="assets",
    )

    @app.get("/{catchall:path}", include_in_schema=False)
    def serve_spa(catchall: str):
        if catchall.startswith(("api/", "api", "docs", "openapi.json", "healthz", "usage")):
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = os.path.join(frontend_dist, catchall)
        if catchall and os.path.exists(file_path) and os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(frontend_dist, "index.html"))
