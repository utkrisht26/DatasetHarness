"""Media-dataset contract extensions (training-ready video delivery).

Covers the Phase-1 contract surface: ``MediaRef`` + ``Record.media``, the
``media`` source channel, media acquisition constraints, ``delivery_layout``,
the video-caption record schema, the Pexels license compatibility entries, and
the delivery-layout registry that ``check_job_spec`` validates against.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from adh.contract import (
    JobSpec,
    MediaRef,
    Record,
    RecordValidationError,
    SourceChannel,
    VerifiabilityTier,
    VerifierResult,
    check_job_spec,
    validate_record,
)
from adh.contract.licenses import is_compatible, normalize_license
from adh.delivery.layouts import (
    hf_layout,
    known_layouts,
    ltx_layout,
    register_layout,
    resolve_layout,
)

ROOT = Path(__file__).resolve().parents[1]
SCHEMAS = ROOT / "schemas"
VIDEO_SCHEMA = json.loads((SCHEMAS / "video_caption_record.schema.json").read_text())
QA_SCHEMA = json.loads((SCHEMAS / "qa_record.schema.json").read_text())

SHA = "a" * 64
CLIP_BYTES = b"\x00\x00\x00\x18ftypisom(fake-clip-payload)"


def _media(**over) -> MediaRef:
    fields = dict(
        role="video",
        path=f"media/{SHA}.mp4",
        sha256=SHA,
        bytes=2048,
        duration_s=6.0,
        width=1080,
        height=1920,
        fps=30.0,
        format="mp4",
        attribution="Pexels: A Videographer (https://www.pexels.com/@a)",
    )
    fields.update(over)
    return MediaRef(**fields)


def _record(media: MediaRef | None = ..., **over) -> Record:
    base = dict(
        record_id="rec-1",
        job_id="job",
        split="train",
        input="a couple dancing outdoors at golden hour",
        output="A couple dances outdoors at golden hour, spinning slowly.",
        source_channel=SourceChannel.MEDIA,
        source_url="https://www.pexels.com/video/12345/",
        source_license="pexels",
        source_sha256=SHA,
        model_id="packager:adh",
        prompt_hash="sha256:" + "b" * 64,
        generated_at=datetime.now(timezone.utc),
        pipeline_step="media-packaging",
        verifiability_tier=VerifiabilityTier.HYBRID,
        verifier_result=VerifierResult(passed=True),
        judge_scores={"helpfulness": 4.0},
        judge_models=["anthropic:a", "openai:b"],
    )
    base.update(over)
    kwargs = {"media": _media()} if media is ... else ({} if media is None else {"media": media})
    return Record(**kwargs, **base)


def _spec(**over) -> JobSpec:
    base = dict(
        name="dance-i2v",
        description="d",
        motivation="m",
        intended_uses=["image-to-video fine-tuning research"],
        verifiability_tier=VerifiabilityTier.HYBRID,
        task_categories=["video-generation"],
        languages=["en"],
        modality="video",
        target_size={"records": 10},
        record_schema_ref="./schemas/video_caption_record.schema.json",
        allowed_licenses=["pexels"],
        generator_models=[],
        judge_config={"models": ["anthropic:a", "openai:b"], "min_score": 3.5},
        splits={"train": 1.0},
        discovery={"channels": ["media"]},
        output_license="cc-by-4.0",
        delivery_layout="ltx",
    )
    base.update(over)
    return JobSpec(**base)


def _stage_clip(tmp_path: Path, sha: str = SHA, ext: str = "mp4") -> Path:
    staged = tmp_path / "media_staging" / "job" / f"{sha}.{ext}"
    staged.parent.mkdir(parents=True, exist_ok=True)
    staged.write_bytes(CLIP_BYTES)
    return staged


# --------------------------------------------------------------------- MediaRef
def test_media_ref_round_trips_through_record():
    record = _record()

    assert record.media is not None and record.media.path == f"media/{SHA}.mp4"
    payload = record.model_dump(mode="json")
    assert payload["media"]["sha256"] == SHA
    assert Record.model_validate(payload).media == record.media


def test_records_default_to_no_media():
    assert _record(media=None).media is None


def test_media_ref_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        _media(codec="h264")


# --------------------------------------------------------------------- schemas
def test_video_schema_validates_media_record():
    assert validate_record(_record(), schema=VIDEO_SCHEMA).record_id == "rec-1"


def test_video_schema_requires_media():
    payload = _record().model_dump(mode="json")
    del payload["media"]

    with pytest.raises(RecordValidationError) as err:
        validate_record(payload, schema=VIDEO_SCHEMA)
    assert "media" in str(err.value)


def test_video_schema_rejects_undersized_media_sha256():
    payload = _record().model_dump(mode="json")
    payload["media"]["sha256"] = "abc"  # pydantic allows; schema demands 64 hex

    with pytest.raises(RecordValidationError):
        validate_record(payload, schema=VIDEO_SCHEMA)


def test_qa_schema_rejects_media_carrying_record():
    """Old text schemas must keep rejecting media rows — loud dead-letter, not silent."""
    with pytest.raises(RecordValidationError):
        validate_record(_record(), schema=QA_SCHEMA)


def test_source_channel_media_round_trips():
    assert SourceChannel("media") is SourceChannel.MEDIA
    assert SourceChannel.MEDIA.value == "media"
    assert _spec().discovery.channels == [SourceChannel.MEDIA]


# --------------------------------------------------------------------- job spec
def test_check_job_spec_accepts_video_spec_with_builtin_layout():
    report = check_job_spec(_spec(), base_dir=ROOT)
    assert report.ok, report.errors


def test_check_job_spec_rejects_unknown_delivery_layout():
    report = check_job_spec(_spec(delivery_layout="bogus"), base_dir=ROOT)
    assert not report.ok
    assert any("delivery_layout" in e and "bogus" in e for e in report.errors)


def test_discovery_constraints_carry_media_filters():
    spec = _spec(
        discovery={
            "channels": ["media"],
            "constraints": {
                "media_min_duration_s": 2,
                "media_max_duration_s": 15,
                "media_orientation": "landscape",
            },
        }
    )
    c = spec.discovery.constraints
    assert (c.media_min_duration_s, c.media_max_duration_s, c.media_orientation) == (
        2,
        15,
        "landscape",
    )


# ------------------------------------------------------------------- registry
def test_known_layouts_include_builtins():
    assert set(known_layouts()) >= {"hf", "ltx"}


def test_resolve_layout_unknown_lists_available():
    with pytest.raises(KeyError, match="available"):
        resolve_layout("nope")


def test_register_layout_extends_the_registry():
    from adh.delivery.layouts import LayoutOutput

    def custom(records, out_dir, staged_root):
        return LayoutOutput()

    register_layout("custom-test", custom)
    try:
        assert resolve_layout("custom-test") is custom
        assert "custom-test" in known_layouts()
    finally:
        from adh.delivery.layouts import _BUILTIN_LAYOUTS

        _BUILTIN_LAYOUTS.pop("custom-test", None)


# ---------------------------------------------------------------- hf layout
def test_hf_layout_matches_mediaref_path_convention(tmp_path):
    staged = _stage_clip(tmp_path)
    out = tmp_path / "dataset"

    result = hf_layout([_record()], out, tmp_path / "media_staging")

    expected = out / "media" / f"{SHA}.mp4"
    assert result.files == (expected,)
    assert result.paths_by_sha == {SHA: f"media/{SHA}.mp4"}
    assert expected.exists()
    assert expected.read_bytes() == CLIP_BYTES
    assert expected.relative_to(out).as_posix() == _record().media.path  # type: ignore[union-attr]
    assert staged.exists()  # staging untouched


def test_hf_layout_fails_closed_when_staged_clip_missing(tmp_path):
    with pytest.raises(RuntimeError, match=SHA[:12]):
        hf_layout([_record()], tmp_path / "ds", tmp_path / "nowhere")


# ---------------------------------------------------------------- ltx layout
def _two_records():
    sha2 = "f" * 64
    second = _record(
        record_id="rec-2",
        split="validation",
        output="The couple dips mid-spin.",
    )
    second = second.model_copy(
        update={
            "media": _media(sha256=sha2, path=f"media/{sha2}.mp4"),
            "source_sha256": sha2,
        }
    )
    return [_record(), second]


def test_ltx_layout_emits_videos_metadata_and_prompts(tmp_path):
    records = _two_records()
    for r in records:
        assert r.media is not None
        _stage_clip(tmp_path, r.media.sha256)
    out = tmp_path / "dataset"

    result = ltx_layout(records, out, tmp_path / "media_staging")
    files = [p.relative_to(out).as_posix() for p in result.files]

    ordered = sorted(records, key=lambda r: (r.split, r.record_id))
    names = [f"{i:05d}_{r.media.sha256[:8]}.mp4" for i, r in enumerate(ordered)]  # type: ignore[union-attr]
    assert files == [f"videos/{n}" for n in names] + ["metadata.jsonl", "prompts.txt"]
    assert result.paths_by_sha == {
        r.media.sha256: f"videos/{n}"  # type: ignore[union-attr]
        for n, r in zip(names, ordered)
    }

    rows = [json.loads(line) for line in (out / "metadata.jsonl").read_text().splitlines()]
    assert rows == [
        {"file_name": f"videos/{n}", "caption": r.output} for n, r in zip(names, ordered)
    ]
    assert (out / "prompts.txt").read_text().splitlines() == [r.output for r in ordered]
    for n in names:
        assert (out / "videos" / n).read_bytes() == CLIP_BYTES


def test_ltx_layout_rebuild_is_idempotent(tmp_path):
    records = _two_records()
    for r in records:
        assert r.media is not None
        _stage_clip(tmp_path, r.media.sha256)

    first = ltx_layout(records, tmp_path / "ds", tmp_path / "media_staging")
    second = ltx_layout(records, tmp_path / "ds", tmp_path / "media_staging")

    assert [p.relative_to(tmp_path / "ds") for p in first.files] == [
        p.relative_to(tmp_path / "ds") for p in second.files
    ]
    assert first.paths_by_sha == second.paths_by_sha
    assert (tmp_path / "ds" / "metadata.jsonl").read_text() == (
        tmp_path / "ds" / "metadata.jsonl"
    ).read_text()


# ------------------------------------------------------------------- licenses
def test_pexels_license_is_permissive_and_normalized():
    assert normalize_license("pexels") == "Pexels-License"
    assert normalize_license("pexels-license") == "Pexels-License"
    assert is_compatible("pexels", "cc-by-4.0")
    assert is_compatible("pexels", "MIT")
    assert not is_compatible("unknown", "cc-by-4.0")  # quarantine rule intact


def test_openverse_short_license_forms_normalize():
    """Openverse ships bare short forms (`by`, `by-sa`, `pdm`) — all must land on canonical names."""
    cases = {
        "cc": "CC-BY-4.0",
        "by": "CC-BY-4.0",
        "by-sa": "CC-BY-SA-4.0",
        "by-nc": "CC-BY-NC-4.0",
        "by-nd": "CC-BY-ND-4.0",
        "by-nc-sa": "CC-BY-NC-SA-4.0",
        "by-nc-nd": "CC-BY-NC-ND-4.0",
        "cc-by": "CC-BY-4.0",
        "cc-by-sa": "CC-BY-SA-4.0",
        "cc-by-nd-4.0": "CC-BY-ND-4.0",
        "cc-by-nc-sa-4.0": "CC-BY-NC-SA-4.0",
        "cc-by-nc-nd-4.0": "CC-BY-NC-ND-4.0",
        "pdm": "public-domain",
        "cc-pdm": "public-domain",
    }
    for raw, want in cases.items():
        assert normalize_license(raw) == want, raw


def test_nc_nd_sharealike_variants_keep_same_license_only_flow():
    """Adding aliases must not widen compatibility: NC/ND/SA variants flow only to themselves."""
    assert is_compatible("CC-BY-NC-ND-4.0", "public-domain") is False
    assert is_compatible("CC-BY-NC-ND-4.0", "CC-BY-NC-ND-4.0") is True
    assert is_compatible("CC-BY-SA-4.0", "CC-BY") is False  # ShareAlike stays ShareAlike


# --------------------------------------------------- media modalities + spec surface
def test_media_modality_membership():
    from adh.contract.enums import MEDIA_MODALITIES, is_media_modality

    assert MEDIA_MODALITIES == frozenset({"video", "image"})
    assert is_media_modality("video")
    assert is_media_modality("Image")
    assert not is_media_modality("text")
    assert not is_media_modality("")
    assert not is_media_modality("code")


def test_resolved_media_types_defaults_infer_from_modality():
    from adh.contract.job_spec import resolved_media_types

    assert resolved_media_types("video") == frozenset({"video"})
    assert resolved_media_types("image") == frozenset({"image", "gif"})
    assert resolved_media_types("text") == frozenset()
    assert resolved_media_types("") == frozenset()


def test_resolved_media_types_explicit_constraints_override():
    from adh.contract.job_spec import DiscoveryConstraints, resolved_media_types

    only_images = DiscoveryConstraints(media_types=["image"])
    assert resolved_media_types("image", constraints=only_images) == frozenset({"image"})

    gifs_only = DiscoveryConstraints(media_types=["GIF"])
    assert resolved_media_types("image", constraints=gifs_only) == frozenset({"gif"})


def test_discovery_allow_unknown_licenses_defaults_off():
    from adh.contract.job_spec import Discovery

    assert Discovery().allow_unknown_licenses is False


# ------------------------------------------- write_dataset media packaging (Phase 6)
def _video_spec(**over) -> JobSpec:
    data = {
        "name": "dance-i2v",
        "description": "dance i2v triplets",
        "motivation": "train LTX image-to-video",
        "intended_uses": ["supervised-fine-tuning"],
        "verifiability_tier": 3,
        "task_categories": ["image-to-video"],
        "languages": ["en"],
        "target_size": {"records": 10},
        "record_schema_ref": "./schemas/video_caption_record.schema.json",
        "allowed_licenses": ["Pexels-License"],
        "output_license": "CC-BY-4.0",
        "modality": "video",
        "delivery_layout": "ltx",
        "output_format": "parquet",
        "generator_models": [],
        "judge_config": {"models": ["anthropic:a", "openai:b"], "min_score": 3.5},
        "splits": {"train": 1.0},
    }
    data.update(over)
    return JobSpec(**data)


def _stage_real(tmp_path, sha: str) -> None:
    staged = tmp_path / "media_staging" / "job"
    staged.mkdir(parents=True, exist_ok=True)
    (staged / f"{sha}.mp4").write_bytes(CLIP_BYTES)


def _second_record() -> Record:
    sha2 = "f" * 64
    rec = _record(record_id="rec-2", output="The couple dips mid-spin.")
    return rec.model_copy(
        update={
            "media": _media(sha256=sha2, path=f"media/{sha2}.mp4", bytes=len(CLIP_BYTES)),
            "source_sha256": sha2,
        }
    )


def test_write_dataset_packages_media_assets_and_rows(tmp_path):
    from adh.delivery import write_dataset

    r1 = _record(media=_media(bytes=len(CLIP_BYTES)))
    r2 = _second_record()
    _stage_real(tmp_path, SHA)
    _stage_real(tmp_path, "f" * 64)
    spec = _video_spec(job_id="job")
    out = tmp_path / "dataset"

    delivery = write_dataset(
        [r1, r2], spec=spec, out_dir=out, staged_media_root=tmp_path / "media_staging"
    )

    video_files = sorted(p.name for p in out.glob("videos/*.mp4"))
    assert len(video_files) == 2
    assert (out / "metadata.jsonl").exists()
    assert (out / "prompts.txt").exists()

    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assets = [row for row in rows if row.get("kind") == "asset"]
    assert {a["sha256"] for a in assets} == {SHA, "f" * 64}
    for a in assets:
        assert a["split"] == "train"
        assert a["format"] == "mp4"
        assert a["bytes"] == len(CLIP_BYTES)
        assert a["duration_s"] == 6.0 and a["width"] == 1080 and a["height"] == 1920
        assert a["record_ids"]
        assert (out / a["path"]).exists(), a["path"]

    names = {p.relative_to(out).as_posix() for p in delivery.files}
    assert {
        "train.parquet",
        *(f"videos/{v}" for v in video_files),
        "metadata.jsonl",
        "prompts.txt",
    } <= names


def test_write_dataset_asset_rows_carry_license_and_attribution(tmp_path):
    """Provenance travels with every shipped asset — the publish gate reads these."""
    from adh.delivery import write_dataset

    r1 = _record(media=_media(bytes=len(CLIP_BYTES)))
    r2 = _second_record()
    _stage_real(tmp_path, SHA)
    _stage_real(tmp_path, "f" * 64)
    out = tmp_path / "dataset"

    write_dataset(
        [r1, r2],
        spec=_video_spec(job_id="job"),
        out_dir=out,
        staged_media_root=tmp_path / "media_staging",
    )

    rows = [json.loads(line) for line in (out / "manifest.jsonl").read_text().splitlines()]
    assets = [row for row in rows if row.get("kind") == "asset"]
    assert len(assets) == 2
    for a in assets:
        assert a["license"] == "pexels"
        assert a["attribution"] == "Pexels: A Videographer (https://www.pexels.com/@a)"


def test_write_dataset_parquet_roundtrip_preserves_media(tmp_path):
    import pyarrow.parquet as pq

    from adh.delivery import write_dataset
    from adh.delivery.writer import decode_parquet_row

    _stage_real(tmp_path, SHA)
    spec = _video_spec(job_id="job")
    out = tmp_path / "dataset"

    write_dataset([_record()], spec=spec, out_dir=out, staged_media_root=tmp_path / "media_staging")

    table = pq.read_table(str(out / "train.parquet"))
    [row] = [decode_parquet_row(r) for r in table.to_pylist()]

    assert row["media"]["sha256"] == SHA
    assert row["media"]["duration_s"] == 6.0


def test_write_dataset_without_staged_root_degrades_loudly(tmp_path):
    from adh.delivery import write_dataset

    delivery = write_dataset([_record()], spec=_video_spec(job_id="job"), out_dir=tmp_path / "ds")

    assert any("staged_media_root" in n for n in delivery.notes)
    kinds = {
        json.loads(line)["kind"]
        for line in (tmp_path / "ds" / "manifest.jsonl").read_text().splitlines()
    }
    assert "asset" not in kinds


# ------------------------------------------------------- publish + evals integration
def test_publish_plan_preserves_media_subdirectories(tmp_path):
    from adh.delivery.publish import publish_dataset

    d = tmp_path / "dataset"
    (d / "videos").mkdir(parents=True)
    (d / "train.jsonl").write_text('{"input": "x", "output": "y"}\n')
    (d / "videos" / "00000_aabbccdd.mp4").write_bytes(b"v")
    (d / "manifest.jsonl").write_text(
        json.dumps({"kind": "shard", "split": "train", "path": "train.jsonl"})
        + "\n"
        + json.dumps(
            {
                "kind": "asset",
                "split": "train",
                "path": "videos/00000_aabbccdd.mp4",
                "sha256": SHA,
                "bytes": 1,
                "format": "mp4",
            }
        )
        + "\n"
    )

    result = publish_dataset(d, repo_id="u/d", license="cc-by-4.0", dry_run=True)

    repo_paths = {u.repo_path for u in result.uploads}
    assert "videos/00000_aabbccdd.mp4" in repo_paths  # NOT flattened to its basename
    assert "train.jsonl" in repo_paths


def test_publish_plan_from_delivery_result_includes_layout_files(tmp_path):
    from adh.delivery import write_dataset
    from adh.delivery.publish import publish_dataset

    _stage_real(tmp_path, SHA)
    delivery = write_dataset(
        [_record()],
        spec=_video_spec(job_id="job"),
        out_dir=tmp_path / "dataset",
        staged_media_root=tmp_path / "media_staging",
    )

    result = publish_dataset(delivery, repo_id="u/d", license="cc-by-4.0", dry_run=True)

    repo_paths = {u.repo_path for u in result.uploads}
    assert any(p.startswith("videos/") and p.endswith(".mp4") for p in repo_paths)
    assert "metadata.jsonl" in repo_paths and "prompts.txt" in repo_paths


def test_evaluate_dataset_reports_media_stats(tmp_path):
    from adh.delivery import evaluate_dataset, write_dataset

    r1 = _record()
    r2 = _second_record()
    _stage_real(tmp_path, SHA)
    _stage_real(tmp_path, "f" * 64)
    out = tmp_path / "dataset"

    write_dataset(
        [r1, r2],
        spec=_video_spec(job_id="job"),
        out_dir=out,
        staged_media_root=tmp_path / "media_staging",
    )

    report = evaluate_dataset(out, schema=VIDEO_SCHEMA)

    assert report.n_media_assets == 2
    assert report.total_media_seconds == pytest.approx(12.0)
    md = report.to_markdown()
    assert "| n_media_assets | 2 |" in md
    assert "| total_media_seconds | 12.0 |" in md


def test_evaluate_dataset_reports_zero_media_for_text_jobs(tmp_path):
    """A plain text dataset carries no media stats rows beyond the zero defaults."""
    from adh.delivery import evaluate_dataset, write_dataset

    text_spec = JobSpec(
        **{
            "name": "t",
            "description": "d",
            "motivation": "m",
            "intended_uses": ["sft"],
            "verifiability_tier": 2,
            "task_categories": ["qa"],
            "languages": ["en"],
            "target_size": {"records": 5},
            "record_schema_ref": "./schemas/qa_record.schema.json",
            "allowed_licenses": ["MIT"],
            "output_license": "MIT",
            "generator_models": [],
            "judge_config": {"models": ["anthropic:a", "openai:b"], "min_score": 3.5},
            "splits": {"train": 1.0},
            "output_format": "jsonl",
        }
    )
    text_rec = Record(
        record_id="r1",
        job_id="t",
        split="train",
        input="q",
        output="a",
        source_channel=SourceChannel.WEB,
        model_id="m",
        prompt_hash="p",
        generated_at=datetime.now(timezone.utc),
        pipeline_step="text-generation",
        verifiability_tier=VerifiabilityTier.SOFT,
    )
    out = tmp_path / "ds"

    write_dataset([text_rec], spec=text_spec, out_dir=out)
    report = evaluate_dataset(out)

    assert report.n_media_assets == 0
    assert report.total_media_seconds == 0.0
    assert "n_media_assets" not in report.to_markdown()
