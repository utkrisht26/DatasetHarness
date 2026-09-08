"""Tests for the three acquisition channels (web / datasets / synthesis)."""

from __future__ import annotations

from datetime import datetime, timezone

from adh.channels import (
    DatasetsChannel,
    FetchContext,
    InMemoryContentFetcher,
    InMemoryDatasetLoader,
    WebChannel,
    handoffs_for,
)
from adh.contract import JobSpec, SourceChannel
from adh.corpus import LocalCorpusStore
from adh.discovery.models import Aspect, CandidateSource, RankedSource, SourceEvaluation


def _spec(minimal_spec_dict) -> JobSpec:
    return JobSpec.model_validate(minimal_spec_dict)


def _ranked(channel, locator, *, aspect="a0", license_=None, via="q") -> RankedSource:
    cand = CandidateSource(
        source_id=f"{aspect}:{channel.value}:{locator}",
        channel=channel,
        locator=locator,
        title=locator,
        aspect_id=aspect,
        license=license_,
        discovered_via=via,
    )
    ev = SourceEvaluation(
        source_id=cand.source_id,
        shape_match=0.8,
        quality=0.8,
        license_ok=bool(license_),
        freshness_ok=True,
        estimated_yield=10,
        score=0.8,
    )
    return RankedSource(candidate=cand, evaluation=ev)


def _ctx(minimal_spec_dict, tmp_path, max_items=50) -> FetchContext:
    return FetchContext(
        job_id="j1",
        store=LocalCorpusStore(tmp_path),
        spec=_spec(minimal_spec_dict),
        max_items_per_source=max_items,
        clock=lambda: datetime.now(timezone.utc),
    )


# ----------------------------------------------------------------- web channel
def test_web_channel_fetches_and_offloads(minimal_spec_dict, tmp_path):
    fetcher = InMemoryContentFetcher(
        pages={"https://docs.rs/p": "# Rust async macros\n\nlong content here"},
        license_hints={"https://docs.rs/p": "Apache-2.0"},
    )
    channel = WebChannel(fetcher=fetcher)
    ctx = _ctx(minimal_spec_dict, tmp_path)
    chunks = channel.fetch(_ranked(SourceChannel.WEB, "https://docs.rs/p"), ctx)
    assert len(chunks) == 1
    assert chunks[0].source_channel == SourceChannel.WEB
    assert chunks[0].source_url == "https://docs.rs/p"
    assert chunks[0].source_license == "Apache-2.0"  # from page hint
    assert ctx.store.get(chunks[0].content_uri).startswith("# Rust async macros")


def test_web_channel_deep_crawl_respects_max_pages(minimal_spec_dict, tmp_path):
    fetcher = InMemoryContentFetcher(
        pages={
            "https://root": "root page content",
            "https://a": "page a content",
            "https://b": "page b content",
        },
        links={"https://root": ["https://a", "https://b"]},
    )
    channel = WebChannel(fetcher=fetcher)
    ctx = _ctx(minimal_spec_dict, tmp_path, max_items=2)
    chunks = channel.fetch(_ranked(SourceChannel.WEB, "https://root"), ctx)
    assert len(chunks) == 2  # capped


def test_web_channel_skips_empty(minimal_spec_dict, tmp_path):
    fetcher = InMemoryContentFetcher(pages={"https://empty": "   "})
    channel = WebChannel(fetcher=fetcher)
    ctx = _ctx(minimal_spec_dict, tmp_path)
    chunks = channel.fetch(_ranked(SourceChannel.WEB, "https://empty"), ctx)
    assert chunks == []


# ----------------------------------------------------------------- datasets channel
def test_datasets_channel_one_chunk_per_record(minimal_spec_dict, tmp_path):
    loader = InMemoryDatasetLoader(
        records={"org/ds": ["record one text", "record two text", "record three text"]}
    )
    channel = DatasetsChannel(loader=loader)
    ctx = _ctx(minimal_spec_dict, tmp_path)
    src = _ranked(SourceChannel.DATASETS, "https://huggingface.co/datasets/org/ds", license_="MIT")
    chunks = channel.fetch(src, ctx)
    assert len(chunks) == 3
    assert all(c.source_channel == SourceChannel.DATASETS for c in chunks)
    assert all(c.derived_from == ("org/ds",) for c in chunks)
    assert all(c.source_license == "MIT" for c in chunks)


def test_datasets_channel_respects_limit(minimal_spec_dict, tmp_path):
    loader = InMemoryDatasetLoader(records={"org/ds": [f"r{i}" for i in range(100)]})
    channel = DatasetsChannel(loader=loader)
    ctx = _ctx(minimal_spec_dict, tmp_path, max_items=5)
    src = _ranked(SourceChannel.DATASETS, "hf://org/ds", license_="MIT")
    chunks = channel.fetch(src, ctx)
    assert len(chunks) == 5


# ----------------------------------------------------------------- synthesis
def test_synthesis_handoffs():
    aspects = (
        Aspect(aspect_id="a0", topic="rust async", rationale="r", keywords=("rust", "async")),
        Aspect(aspect_id="a1", topic="sql", rationale="r", keywords=("sql",)),
    )
    handoffs = handoffs_for("j1", aspects)
    assert len(handoffs) == 2
    assert handoffs[0].topic == "rust async"
    assert handoffs[0].keywords == ("rust", "async")
    row = handoffs[0].to_manifest_row()
    assert row["source_channel"] == "synthesis"
