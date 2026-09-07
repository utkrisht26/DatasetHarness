"""Engine B video path — deterministic MediaPackager, sha-keyed novelty/dedup.

Offline end-to-end: hand-built MP4s staged content-addressed (the exact bytes the
MEDIA channel will produce), fixed seeds pointing at them, tier-3 verification
through the real media-probe hook. No model, no network.
"""

from __future__ import annotations

import json
import struct

import pytest

from adh.channels.media.staging import MediaStagingStore
from adh.delivery.evals import assert_publishable
from adh.contract import JobSpec, SourceChannel
from adh.generation import MediaPackager, SeedPoint
from adh.generation.models import make_candidate_id
from adh.generation.seed import NoveltyGate
from adh.orchestration.engine_b import run_engine_b
from adh.verification import Verdict


def _mp4(duration_s: float = 6.0, width: int = 1920, height: int = 1080) -> bytes:
    """Minimal but structurally-real MP4 (same layout test_media_probe pins)."""

    def box(btype: bytes, *payloads: bytes) -> bytes:
        body = b"".join(payloads)
        return struct.pack(">I", 8 + len(body)) + btype + body

    mvhd = bytearray(100)
    mvhd[12:16] = struct.pack(">I", 1000)
    mvhd[16:20] = struct.pack(">I", int(duration_s * 1000))
    tkhd = bytearray(84)
    tkhd[76:80] = struct.pack(">I", width * 65536)
    tkhd[80:84] = struct.pack(">I", height * 65536)
    return (
        box(b"ftyp", b"isom" + struct.pack(">I", 512) + b"isomiso2mp41")
        + box(b"moov", box(b"mvhd", bytes(mvhd)), box(b"trak", box(b"tkhd", bytes(tkhd))))
        + box(b"mdat", b"\x00" * 32)
    )


def _stage(tmp_path, data: bytes, meta: dict, job="job"):
    store = MediaStagingStore(tmp_path / "media_staging")
    return store.stage(job, data, ext="mp4", meta=meta)


def _seed_for(clip, *, seed_id="s1", topic="a couple dancing on the beach") -> SeedPoint:
    return SeedPoint(
        seed_id=seed_id,
        job_id="job",
        aspect_id="a1",
        topic=topic,
        keywords=("dance",),
        persona="",
        difficulty="",
        source_channel=SourceChannel.MEDIA,
        grounding_uri=str(clip.path),
        source_url="https://www.pexels.com/video/1/",
        source_license="Pexels-License",
        source_sha256=clip.sha256,
    )


def _video_spec(minimal_spec_dict, **over) -> JobSpec:
    data = {
        **minimal_spec_dict,
        "name": "dance-i2v",
        "description": "dance i2v triplets",
        "motivation": "train LTX image-to-video",
        "modality": "video",
        "verifiability_tier": 3,
        "task_categories": ["image-to-video"],
        "target_size": {"records": 10},
        "record_schema_ref": "./schemas/video_caption_record.schema.json",
        "allowed_licenses": ["Pexels-License"],
        "output_license": "CC-BY-4.0",
        "generator_models": [],
        "delivery_layout": "ltx",
        **over,
    }
    return JobSpec(**data)


class _FixedSeeds:
    def __init__(self, seeds):
        self._seeds = list(seeds)

    def plan(self, job_id, **kw):
        return list(self._seeds)


# ------------------------------------------------------------------ packager unit
def test_packager_builds_media_candidate_from_staged_clip(tmp_path, minimal_spec_dict):
    meta = {
        "title": "Couple dancing on the beach at sunset",
        "tags": ["dance", "sunset"],
        "license": "Pexels-License",
        "url": "https://www.pexels.com/video/1/",
        "duration_s": 6.0,
        "width": 1920,
        "height": 1080,
        "format": "mp4",
    }
    clip = _stage(tmp_path, _mp4(), meta)
    spec = _video_spec(minimal_spec_dict)

    [cand] = MediaPackager().generate(_seed_for(clip), spec=spec)

    assert cand.media.path == f"media/{clip.sha256}.mp4"
    assert cand.media.sha256 == clip.sha256
    assert cand.media.bytes == clip.path.stat().st_size
    assert cand.media.duration_s == 6.0
    assert cand.media.width == 1920 and cand.media.height == 1080
    assert cand.staged_path == str(clip.path.resolve())
    assert cand.output.startswith("Couple dancing on the beach")
    assert str(clip.sha256[:8]) in cand.input  # scene descriptor carries asset identity
    assert cand.model_id == "packager:adh"
    assert cand.pipeline_step == "media-packaging"
    assert cand.source_channel == SourceChannel.MEDIA


def test_packager_skips_missing_clip_without_raising(tmp_path, minimal_spec_dict):
    ghost = type("C", (), {"path": tmp_path / "gone.mp4", "sha256": "0" * 64})()
    spec = _video_spec(minimal_spec_dict)

    assert MediaPackager().generate(_seed_for(ghost), spec=spec) == []


PNG_BYTES = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x40\x00\x00\x00\xf0"
    b"\x08\x02\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_packager_roles_assets_from_sidecar_media_type(tmp_path, minimal_spec_dict):
    meta = {
        "title": "A mountain lake at dawn",
        "tags": ["landscape"],
        "license": "CC-BY-4.0",
        "url": "https://openweb.example/photo/1",
        "attribution": "A mountain lake at dawn",
        "width": 320,
        "height": 240,
        "media_type": "image",
        "format": "png",
    }
    store = MediaStagingStore(tmp_path / "media_staging")
    img = store.stage("job", PNG_BYTES, ext="png", meta=meta)
    seed = _seed_for(img, topic="mountain lakes at dawn")

    [cand] = MediaPackager().generate(seed, spec=_video_spec(minimal_spec_dict))

    assert cand.media.role == "image"  # role follows the sidecar, not a video default
    assert cand.media.path.endswith(".png")
    assert cand.media.duration_s is None
    assert f"asset {img.sha256[:8]}" in cand.input  # generic descriptor word


# ------------------------------------------------------- sha-keyed gates (unit pins)
def test_novelty_gate_treats_media_sha_as_identity():
    gate = NoveltyGate()

    assert gate.is_novel("a" * 64) is True
    assert gate.is_novel("a" * 64) is False  # same asset twice -> not novel
    assert gate.is_novel("b" * 64) is True


def test_video_candidates_dedup_on_sha_not_caption(minimal_spec_dict, repo_root, tmp_path):
    """Two different clips with an identical caption are training signal, not dups."""
    from adh.orchestration.engine_b import _curate

    spec = _video_spec(minimal_spec_dict)
    caption = "A couple dancing on the beach at golden hour"
    cands = [
        _video_candidate(minimal_spec_dict, sha="a" * 64, caption=caption),
        _video_candidate(minimal_spec_dict, sha="b" * 64, caption=caption),
    ]
    kept = [(c, Verdict(True, "jury", "ok")) for c in cands]

    records, _, dead, duplicates, _ = _curate(
        kept, spec=spec, schema=None, schema_base_dir=repo_root
    )

    assert duplicates == 0
    assert len(records) == 2
    assert {r.media.sha256 for r in records} == {"a" * 64, "b" * 64}
    assert not dead


def _video_candidate(minimal_spec_dict, *, sha, caption):
    from adh.contract import MediaRef, VerifiabilityTier
    from adh.generation import Candidate

    return Candidate(
        candidate_id=make_candidate_id("job", "a1", f"scene {sha[:8]}", caption),
        job_id="job",
        aspect_id="a1",
        verifiability_tier=VerifiabilityTier.HYBRID,
        input=f"a couple dancing: clip {sha[:8]}",
        output=caption,
        source_channel=SourceChannel.MEDIA,
        source_license="Pexels-License",
        media=MediaRef(path=f"media/{sha}.mp4", sha256=sha, bytes=2048, format="mp4"),
        staged_path="/unused/" + sha + ".mp4",
        pipeline_step="media-packaging",
    )


# ------------------------------------------------------------------- offline e2e
def test_engine_b_video_end_to_end_offline(minimal_spec_dict, repo_root, tmp_path):
    meta_a = {
        "title": "Couple dancing on the beach at sunset",
        "tags": ["dance"],
        "license": "Pexels-License",
        "url": "https://www.pexels.com/video/1/",
        "duration_s": 6.0,
        "width": 1920,
        "height": 1080,
        "format": "mp4",
    }
    meta_b = dict(
        meta_a,
        title="Dog catching a frisbee in slow motion",
        url="https://www.pexels.com/video/2/",
        duration_s=4.5,  # matches the staged clip's real probed length
    )
    clip_a = _stage(tmp_path, _mp4(), meta_a)
    clip_b = _stage(tmp_path, _mp4(duration_s=4.5), meta_b)
    spec = _video_spec(minimal_spec_dict)
    seeds = [
        _seed_for(clip_a, seed_id="s1"),
        _seed_for(clip_b, seed_id="s2", topic="a dog catching a frisbee"),
        _seed_for(clip_a, seed_id="s3"),  # same asset re-surfaced -> novelty drops it
    ]

    result = run_engine_b(
        spec,
        work_dir=tmp_path,
        seed_engine=_FixedSeeds(seeds),
        schema_base_dir=repo_root,
        resume=False,
    )

    assert result.exit_reason == "completed"
    assert result.n_records == 2
    assert result.counts["dropped_novelty"] == 1
    assert result.counts["dropped_verifier"] == 0  # real clips pass the media probe
    for rec in result.records:
        assert rec.media is not None
        assert rec.source_channel == SourceChannel.MEDIA
    assert result.quality_report.schema_validation_pass_rate == 1.0

    # ---- the delivered bundle is trainer-ready: clips shipped, aligned metadata,
    # asset manifest rows, croissant distribution, quality-card media stats
    video_files = sorted(p.name for p in (result.dataset_dir / "videos").glob("*.mp4"))
    assert len(video_files) == 2
    assert {sha[:8] for sha in (clip_a.sha256, clip_b.sha256)} == {
        name.split("_", 1)[1].removesuffix(".mp4") for name in video_files
    }

    meta_rows = [
        json.loads(line)
        for line in (result.dataset_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [row["file_name"] for row in meta_rows] == [f"videos/{n}" for n in video_files]
    assert {row["caption"] for row in meta_rows} == {
        "Couple dancing on the beach at sunset, dance",
        "Dog catching a frisbee in slow motion, dance",
    }
    prompts = (result.dataset_dir / "prompts.txt").read_text(encoding="utf-8").splitlines()
    assert prompts == [row["caption"] for row in meta_rows]

    manifest_rows = [
        json.loads(line)
        for line in (result.dataset_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    asset_rows = [row for row in manifest_rows if row.get("kind") == "asset"]
    assert {row["sha256"] for row in asset_rows} == {clip_a.sha256, clip_b.sha256}
    assert sum(row["duration_s"] for row in asset_rows) == pytest.approx(10.5)

    croissant = json.loads((result.dataset_dir / "croissant.json").read_text(encoding="utf-8"))
    dist_ids = {entry["@id"] for entry in croissant["distribution"]}
    assert dist_ids >= {f"videos/{n}" for n in video_files} | {"metadata.jsonl", "prompts.txt"}

    assert result.quality_report.n_media_assets == 2
    assert result.quality_report.total_media_seconds == pytest.approx(10.5)
    assert_publishable(result.quality_report, min_records=2)
