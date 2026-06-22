"""The ``datasets`` channel — reuse existing datasets (OUTLINE §1.6).

Usually the highest-yield path: download a licensed dataset and normalize it into the
corpus instead of re-scraping (RESEARCH §DISCOVERY, reuse > crawl). A
:class:`DatasetLoader` port abstracts loading; :class:`InMemoryDatasetLoader` runs
offline, :class:`HfDatasetLoader` lazy-imports the ``datasets`` library in production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Protocol, runtime_checkable

from adh.contract import SourceChannel
from adh.corpus import CorpusChunk, offload
from adh.discovery.models import RankedSource

from .base import FetchContext, make_chunk_id


@runtime_checkable
class DatasetLoader(Protocol):
    def load(self, dataset_id: str, *, limit: int) -> Iterable[str]:
        """Yield up to ``limit`` records, each rendered as a text blob."""
        ...


@dataclass
class InMemoryDatasetLoader:
    """Deterministic loader over a seeded record set (offline / tests)."""

    records: dict[str, list[str]] = field(default_factory=dict)  # dataset_id -> [text, ...]

    def load(self, dataset_id: str, *, limit: int) -> Iterable[str]:
        return list(self.records.get(dataset_id, []))[:limit]


@dataclass
class HfDatasetLoader:
    """Live loader over the HF ``datasets`` library (optional extra ``adh[datasets]``)."""

    split: str = "train"
    text_fields: tuple[str, ...] = ("text", "input", "output", "question", "answer")

    def load(self, dataset_id: str, *, limit: int) -> Iterable[str]:  # pragma: no cover
        try:
            from datasets import load_dataset  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "the 'datasets' library is required for the live datasets channel; "
                "install with: pip install 'adh[datasets]'"
            ) from exc

        ds = load_dataset(dataset_id, split=self.split, streaming=True)
        out: list[str] = []
        for row in ds:
            parts = [str(row[f]) for f in self.text_fields if f in row and row[f] is not None]
            blob = "\n".join(parts) if parts else str(row)
            out.append(blob)
            if len(out) >= limit:
                break
        return out


def _dataset_id_from_locator(locator: str) -> str:
    # https://huggingface.co/datasets/<id>  |  hf://<id>  |  <id>
    if "huggingface.co/datasets/" in locator:
        return locator.split("huggingface.co/datasets/", 1)[1].strip("/")
    if locator.startswith("hf://"):
        return locator[len("hf://") :].strip("/")
    return locator


@dataclass
class DatasetsChannel:
    """Acquire an existing dataset's records and offload each as a corpus chunk."""

    loader: DatasetLoader
    name: SourceChannel = SourceChannel.DATASETS

    def fetch(self, source: RankedSource, ctx: FetchContext) -> list[CorpusChunk]:
        dataset_id = _dataset_id_from_locator(source.candidate.locator)
        license_ = source.candidate.license or "unknown"
        chunks: list[CorpusChunk] = []
        for text in self.loader.load(dataset_id, limit=ctx.max_items_per_source):
            if not text.strip():
                continue
            chunks.append(
                offload(
                    ctx.store,
                    job_id=ctx.job_id,
                    aspect_id=source.candidate.aspect_id,
                    content=text,
                    source_channel=SourceChannel.DATASETS,
                    source_url=source.candidate.locator,
                    source_license=license_,
                    chunk_id=make_chunk_id(ctx.job_id, source.candidate.aspect_id, text),
                    derived_from=(dataset_id,),
                    discovered_via=source.candidate.discovered_via,
                    fetched_at=ctx.clock(),
                )
            )
        return chunks
