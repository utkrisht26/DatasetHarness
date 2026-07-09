"""HTTP gateway — one thin app, per-domain routers under ``/api/<domain>/*``.

Locked decision 5: gateway contract now, split when forced. The frontend talks to
one origin; lifting a prefix (``/api/crawl``, ``/api/runs``) to its own process
later is a config flip, not an API change.
"""

from __future__ import annotations

from .app import create_app

__all__ = ["create_app"]
