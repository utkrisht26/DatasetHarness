"""Small security primitives shared by the HTTP gateway and its CLI entry point."""

from __future__ import annotations

import ipaddress
import secrets
from collections.abc import Iterable

from starlette.responses import JSONResponse

API_KEY_ENV = "ADH_API_KEY"
DEFAULT_PUBLIC_PATHS = frozenset({"/api/healthz"})


def is_loopback_host(host: str) -> bool:
    """Whether ``host`` confines the server to the local machine."""
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


class BearerTokenMiddleware:
    """Require one configured bearer token for protected HTTP API paths.

    This is deliberately a narrow deployment perimeter, not a replacement for
    tenant-aware authorization. Token comparisons are constant-time and the
    configured secret is never included in a response or log message.
    """

    def __init__(
        self,
        app,
        *,
        token: str,
        public_paths: Iterable[str] = DEFAULT_PUBLIC_PATHS,
    ) -> None:
        if not token:
            raise ValueError("bearer token must not be empty")
        self.app = app
        self._token = token.encode("utf-8")
        self._public_paths = frozenset(public_paths)

    async def __call__(self, scope, receive, send) -> None:
        if not self._requires_authentication(scope):
            await self.app(scope, receive, send)
            return

        authorization = dict(scope.get("headers", ())).get(b"authorization", b"")
        scheme, separator, credential = authorization.partition(b" ")
        authenticated = (
            bool(separator)
            and scheme.lower() == b"bearer"
            and bool(credential)
            and secrets.compare_digest(credential, self._token)
        )
        if authenticated:
            await self.app(scope, receive, send)
            return

        response = JSONResponse(
            {"detail": "invalid or missing bearer token"},
            status_code=401,
            headers={"WWW-Authenticate": "Bearer"},
        )
        await response(scope, receive, send)

    def _requires_authentication(self, scope) -> bool:
        if scope["type"] != "http" or scope.get("method") == "OPTIONS":
            return False
        path = scope.get("path", "")
        is_api_path = path == "/api" or path.startswith("/api/")
        return is_api_path and path not in self._public_paths
