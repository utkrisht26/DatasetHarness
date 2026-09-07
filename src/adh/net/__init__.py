"""Shared network plumbing — polite, SSRF-guarded fetching for open-internet sources."""

from .httpfetch import (
    DEFAULT_ASSET_CONTENT_TYPES,
    OversizedAssetError,
    PoliteFetcher,
    SSRFPolicy,
    UnsupportedContentType,
    UrlBlocked,
    is_blocked_ip,
    resolve_hostname,
)

__all__ = [
    "DEFAULT_ASSET_CONTENT_TYPES",
    "OversizedAssetError",
    "PoliteFetcher",
    "SSRFPolicy",
    "UnsupportedContentType",
    "UrlBlocked",
    "is_blocked_ip",
    "resolve_hostname",
]
