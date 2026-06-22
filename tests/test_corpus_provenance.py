"""Tests for the corpus store, context offload, and provenance/lineage."""

from __future__ import annotations

from datetime import datetime, timezone

from adh.contract import SourceChannel
from adh.corpus import LocalCorpusStore, offload
from adh.corpus.models import CorpusChunk
from adh.provenance import (
    approx_token_count,
    build_lineage,
    sha256_hex,
    sha256_prefixed,
    write_lineage,
)


def test_store_is_content_addressed(tmp_path):
    store = LocalCorpusStore(tmp_path)
    uri1 = store.put("job1", "hello world")
    uri2 = store.put("job1", "hello world")  # identical -> same blob
    assert uri1 == uri2
    assert store.exists(uri1)
    assert store.get(uri1) == "hello world"


def test_store_size_mb_grows(tmp_path):
    store = LocalCorpusStore(tmp_path)
    assert store.size_mb("job1") == 0.0
    store.put("job1", "x" * 1024)
    assert store.size_mb("job1") > 0.0


def test_offload_keeps_only_preview(tmp_path):
    store = LocalCorpusStore(tmp_path)
    big = "word " * 1000
    chunk = offload(
        store,
        job_id="job1",
        aspect_id="a0",
        content=big,
        source_channel=SourceChannel.WEB,
        source_url="https://x.test/p",
        source_license="MIT",
        chunk_id="c1",
    )
    assert len(chunk.text_preview) <= 280
    assert chunk.n_tokens == 1000
    assert store.get(chunk.content_uri) == big  # full text retrievable from store
    assert chunk.source_sha256 == sha256_hex(big)


def test_sha_and_tokens():
    assert sha256_prefixed("abc").startswith("sha256:")
    assert approx_token_count("a b c") == 3


def test_manifest_row_round_trip():
    ch = CorpusChunk(
        chunk_id="c1",
        job_id="j1",
        aspect_id="a0",
        source_channel=SourceChannel.DATASETS,
        source_url="https://huggingface.co/datasets/x",
        source_license="MIT",
        source_sha256="deadbeef",
        content_uri="/tmp/x.txt",
        text_preview="preview",
        n_tokens=42,
        fetched_at=datetime.now(timezone.utc),
        derived_from=("x",),
        discovered_via="q",
    )
    row = ch.to_manifest_row()
    assert row["kind"] == "chunk"
    back = CorpusChunk.from_manifest_row(row, text_preview="preview")
    assert back.chunk_id == ch.chunk_id
    assert back.source_channel == SourceChannel.DATASETS
    assert back.n_tokens == 42
    assert back.derived_from == ("x",)


def _chunk(channel, url, sha, aspect="a0"):
    return CorpusChunk(
        chunk_id=f"c-{sha}",
        job_id="j1",
        aspect_id=aspect,
        source_channel=channel,
        source_url=url,
        source_license="CC-BY-4.0",
        source_sha256=sha,
        content_uri=f"/tmp/{sha}.txt",
        text_preview="p",
        n_tokens=10,
        fetched_at=datetime.now(timezone.utc),
    )


def test_build_lineage_prov_shape(tmp_path):
    chunks = [
        _chunk(SourceChannel.WEB, "https://a.test/1", "aaaa1111"),
        _chunk(SourceChannel.DATASETS, "https://huggingface.co/datasets/x", "bbbb2222", "a1"),
    ]
    lineage = build_lineage("j1", chunks, exit_reason="completed")
    assert lineage["job_id"] == "j1"
    assert lineage["exit_reason"] == "completed"
    # chunk entities + source entities
    entity_ids = {e["id"] for e in lineage["entities"]}
    assert "c-aaaa1111" in entity_ids
    assert any(e["type"] == "source" for e in lineage["entities"])
    assert lineage["counts"]["agents"] == 2  # web + datasets channels
    # chunk wasDerivedFrom a source entity
    chunk_ent = next(e for e in lineage["entities"] if e["id"] == "c-aaaa1111")
    assert chunk_ent["wasDerivedFrom"]
    path = write_lineage(tmp_path / "lineage.json", lineage)
    assert path.exists()


def test_lineage_synthesis_chunk_has_no_source_entity():
    chunk = _chunk(SourceChannel.SYNTHESIS, None, "cccc3333")
    lineage = build_lineage("j1", [chunk])
    assert all(e["type"] != "source" for e in lineage["entities"])
