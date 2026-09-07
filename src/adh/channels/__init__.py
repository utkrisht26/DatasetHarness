"""The acquisition channels: media > datasets (reuse) > web (crawl) > synthesis."""

from __future__ import annotations

from .base import Channel, FetchContext, make_chunk_id
from .datasets import (
    DatasetLoader,
    DatasetsChannel,
    HfDatasetLoader,
    InMemoryDatasetLoader,
)
from .media import (
    CompositeMediaProvider,
    InMemoryMediaProvider,
    MediaChannel,
    MediaHit,
    MediaProvider,
)
from .synthesis import handoffs_for
from .web import (
    ContentFetcher,
    Crawl4AIContentFetcher,
    FetchedPage,
    InMemoryContentFetcher,
    WebChannel,
)

__all__ = [
    "Channel",
    "FetchContext",
    "make_chunk_id",
    # web
    "WebChannel",
    "ContentFetcher",
    "InMemoryContentFetcher",
    "Crawl4AIContentFetcher",
    "FetchedPage",
    # datasets
    "DatasetsChannel",
    "DatasetLoader",
    "InMemoryDatasetLoader",
    "HfDatasetLoader",
    # media
    "MediaChannel",
    "MediaProvider",
    "InMemoryMediaProvider",
    "CompositeMediaProvider",
    "MediaHit",
    # synthesis
    "handoffs_for",
]


def __getattr__(name: str):
    # PexelsMediaProvider needs httpx (extra ``adh[media]``) — lazy like the
    # other optional-extra adapters.
    if name == "PexelsMediaProvider":
        from .media.pexels import PexelsMediaProvider

        return PexelsMediaProvider
    raise AttributeError(name)
