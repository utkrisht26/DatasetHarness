"""Search ports — web meta-search and dataset registries."""

from __future__ import annotations

from .provider import (
    DatasetHit,
    DatasetRegistry,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
    SearchHit,
    SearchProvider,
)

__all__ = [
    "SearchProvider",
    "DatasetRegistry",
    "SearchHit",
    "DatasetHit",
    "InMemorySearchProvider",
    "InMemoryDatasetRegistry",
]
