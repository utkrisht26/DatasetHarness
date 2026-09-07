"""MEDIA acquisition channel — open-internet media download and staging."""

from __future__ import annotations

from .channel import MediaChannel
from .composite import CompositeMediaProvider
from .provider import InMemoryMediaProvider, MediaHit, MediaProvider
from .staging import MediaStagingStore, StagedClip, StorageBudgetExceeded

__all__ = [
    "MediaChannel",
    "MediaProvider",
    "InMemoryMediaProvider",
    "CompositeMediaProvider",
    "MediaHit",
    "MediaStagingStore",
    "StagedClip",
    "StorageBudgetExceeded",
]

# Live providers lazy-import their optional transports (httpx behind the media
# extra; yt-dlp behind webvideo) — keep them out of the import path unless
# explicitly requested. direct_web/openverse themselves are stdlib-only but
# stay here for symmetry with the runtime's guarded wiring.
_LIVE_PROVIDERS = {
    "PexelsMediaProvider": ".pexels",
    "OpenverseImageProvider": ".openverse",
    "DirectWebMediaProvider": ".direct_web",
    "YtDlpMediaProvider": ".ytdlp",
}


def __getattr__(name: str):
    module = _LIVE_PROVIDERS.get(name)
    if module is not None:
        from importlib import import_module

        return getattr(import_module(module, __package__), name)
    raise AttributeError(name)
