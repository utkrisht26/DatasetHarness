"""Content-addressed media staging — budgeted, atomic, idempotent.

Tests drive the store through all three source shapes (local path, bytes,
byte-chunk stream — the shape an httpx download arrives as) and pin the two
budget enforcement points: pre-flight rejection and mid-stream abort.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from adh.channels.media.staging import (
    MediaStagingStore,
    StorageBudgetExceeded,
)


def _blob(size: int = 2048) -> bytes:
    return bytes(range(256)) * (size // 256)


def _store(tmp_path, **kw) -> MediaStagingStore:
    return MediaStagingStore(tmp_path / "media_staging", **kw)


def _write_src(tmp_path, data: bytes, name: str = "clip.mp4"):
    p = tmp_path / name
    p.write_bytes(data)
    return p


def test_stage_copies_local_file_content_addressed(tmp_path):
    data = _blob()
    store = _store(tmp_path)

    clip = store.stage("job1", _write_src(tmp_path, data))

    sha = hashlib.sha256(data).hexdigest()
    assert clip.path.parent == tmp_path / "media_staging" / "job1"
    assert clip.path.name == f"{sha}.mp4"  # extension inherited from the source
    assert clip.sha256 == sha
    assert clip.bytes == len(data)
    assert clip.path.read_bytes() == data


def test_stage_accepts_bytes_and_chunk_streams(tmp_path):
    data = _blob(1024)
    store = _store(tmp_path)

    from_bytes = store.stage("j", data, ext="mp4")
    from_chunks = store.stage("j", iter([data[:512], data[512:]]), ext="mp4")

    assert from_bytes.sha256 == hashlib.sha256(data).hexdigest()
    assert from_chunks.path == from_bytes.path
    assert from_chunks.bytes == len(data)


def test_restaging_identical_content_is_idempotent(tmp_path):
    data = _blob()
    store = _store(tmp_path)
    first = store.stage("job1", _write_src(tmp_path, data))
    again = store.stage("job1", _write_src(tmp_path, data, name="duplicate.mp4"))

    assert again == first
    # exactly the clip + its sidecar — no temp litter, no duplicate copies
    assert sorted(p.name for p in first.path.parent.iterdir()) == [
        f"{first.sha256}.mp4",
        f"{first.sha256}.mp4.meta.json",
    ]


def test_preflight_rejects_oversized_known_size_before_writing(tmp_path):
    data = _blob(4096)
    store = _store(tmp_path, max_bytes=len(data) - 1)

    with pytest.raises(StorageBudgetExceeded, match="budget"):
        store.stage("job1", _write_src(tmp_path, data))

    assert not list((tmp_path / "media_staging").rglob("*.mp4"))


def test_mid_stream_abort_when_chunked_size_only_reveals_itself(tmp_path):
    data = _blob(1024)
    store = _store(tmp_path, max_bytes=1536)

    with pytest.raises(StorageBudgetExceeded, match="budget"):
        store.stage("job1", iter([data, data, data]), ext="mp4")

    # partial temp file must be cleaned up, nothing staged
    leftovers = [p for p in (tmp_path / "media_staging").rglob("*") if p.is_file()]
    assert not leftovers


def test_budget_counts_previously_staged_jobs(tmp_path):
    data = _blob(1024)
    store = _store(tmp_path, max_bytes=1536)
    store.stage("job1", _write_src(tmp_path, data))

    with pytest.raises(StorageBudgetExceeded):
        store.stage("job2", _write_src(tmp_path, data, name="b.mp4"))


def test_sidecar_meta_written_and_read_back(tmp_path):
    data = _blob()
    store = _store(tmp_path)
    meta = {"title": "Couple dancing", "tags": ["dance"], "license": "Pexels-License"}

    clip = store.stage("job1", _write_src(tmp_path, data), meta=meta)

    sidecar = clip.path.with_name(clip.path.name + ".meta.json")
    assert json.loads(sidecar.read_text()) == meta
    assert store.read_meta("job1", clip.sha256) == meta


def test_read_meta_returns_empty_dict_when_missing(tmp_path):
    assert _store(tmp_path).read_meta("jobX", "0" * 64) == {}


def test_staged_bytes_roundtrip(tmp_path):
    data = _blob()
    store = _store(tmp_path)

    clip = store.stage("job1", _write_src(tmp_path, data))

    assert store.staged_bytes(clip) == data
