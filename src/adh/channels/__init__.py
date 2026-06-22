"""The three acquisition channels: datasets (reuse) > web (crawl) > synthesis."""

from __future__ import annotations

from .base import Channel, FetchContext, make_chunk_id
from .datasets import (
    DatasetLoader,
    DatasetsChannel,
    HfDatasetLoader,
    InMemoryDatasetLoader,
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
    # synthesis
    "handoffs_for",
]
