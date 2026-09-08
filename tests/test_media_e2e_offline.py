"""Whole-spine offline acceptance — MEDIA acquisition → Engine B → delivery → publish gate.

One mixed-license image job walks the entire pipeline: the channel stages a
CC-BY-4.0 photo and an open-web photo whose license nobody declared (spec opt-in),
provenance rides every artifact, quality evals count the unknown asset, and
PUBLIC publication refuses until explicitly overridden. No model, no network.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from adh import cli
from adh.channels import FetchContext
from adh.channels.media import InMemoryMediaProvider, MediaChannel, MediaHit
from adh.contract import JobSpec, SourceChannel
from adh.corpus.models import CorpusChunk
from adh.delivery.evals import evaluate_dataset
from adh.delivery.publish import PublishRefusedError, publish_dataset
from adh.discovery.models import CandidateSource, RankedSource, SourceEvaluation
from adh.generation import SeedPoint


def _png(width: int = 640, height: int = 480, marker: int = 0) -> bytes:
    """Hand-built structurally-real PNG; the probe reads IHDR dims."""
    ihdr = struct.pack(">II", width, height)
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + ihdr
        + bytes([8, 2, 0, marker])  # bit depth 8, truecolor; marker varies the CRC byte
        + struct.pack(">I", 0)
        + b"IEND"
        + struct.pack(">I", 0xAE426082)
    )


def _hit(**over) -> MediaHit:
    fields = dict(
        media_id="cc",
        page_url="https://openverse.org/photos/cc",
        download_url="https://cdn.example/cc.png",
        # descriptive alt-text style: long enough for the heuristic jury's
        # length plateau and overlapping the seed topic's vocabulary
        title="A mountain lake at dawn reflecting the surrounding snowy peaks",
        tags=("landscape", "mountain"),
        width=640,
        height=480,
        license="CC-BY-4.0",
        attribution="Ansel Adams (Openverse)",
        media_type="image",
        content=_png(marker=1),
    )
    fields.update(over)
    return MediaHit(**fields)


def _spec(minimal_spec_dict, **over) -> JobSpec:
    data = {
        **minimal_spec_dict,
        "name": "open-images",
        "description": "mixed-license landscape images with captions",
        "motivation": "image-captioning finetuning from the open web",
        "modality": "image",
        "verifiability_tier": 3,
        "task_categories": ["image-captioning"],
        "target_size": {"records": 10},
        "record_schema_ref": "./schemas/image_caption_record.schema.json",
        "allowed_licenses": ["CC-BY-4.0"],
        "output_license": "CC-BY-4.0",
        "generator_models": [],
        "judge_config": {"models": ["anthropic:a", "openai:b"], "min_score": 3.5},
        "delivery_layout": "hf",
        "discovery": {"channels": ["media"], "allow_unknown_licenses": True},
        **over,
    }
    return JobSpec(**data)


class _NullStore:
    def put(self, job_id: str, content: str) -> str:  # pragma: no cover - guard
        raise AssertionError("media channel must not offload text to the corpus store")

    def size_mb(self, job_id: str) -> float:
        return 0.0


class _Governor:
    def observe(self, aspect_id: str, text: str) -> None:
        pass

    def should_stop(self, aspect_id: str) -> bool:
        return False


@dataclass
class FakeHubClient:
    """Records every Hub call; returns deterministic fake URLs."""

    calls: list[tuple[str, dict]] = field(default_factory=list)

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str = "dataset",
        private: bool = True,
        exist_ok: bool = True,
    ) -> str:
        self.calls.append(("create_repo", {"repo_id": repo_id, "private": private}))
        return f"https://huggingface.co/{repo_type}s/{repo_id}"

    def upload_file(
        self,
        *,
        path_or_fileobj,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "dataset",
        commit_message: str = "",
    ) -> str:
        self.calls.append(("upload_file", {"path_in_repo": path_in_repo}))
        return f"https://huggingface.co/{repo_type}s/{repo_id}/blob/main/{path_in_repo}"


@dataclass
class _ChunksAsSeeds:
    """Bridges MediaChannel output into Engine B seed points (offline glue)."""

    chunks: list[CorpusChunk]

    def plan(self, job_id: str, **kwargs) -> list[SeedPoint]:
        return [
            SeedPoint(
                seed_id=f"s{i}",
                job_id=job_id,
                aspect_id=ch.aspect_id,
                topic="mountain lakes at dawn",
                keywords=("lake",),
                persona="",
                difficulty="",
                source_channel=ch.source_channel,
                grounding_uri=ch.content_uri,
                source_url=str(ch.source_url) if ch.source_url else None,
                source_license=ch.source_license,
                source_sha256=ch.source_sha256,
            )
            for i, ch in enumerate(self.chunks, start=1)
        ]


def test_mixed_license_image_job_end_to_end(minimal_spec_dict, repo_root, tmp_path):
    spec = _spec(minimal_spec_dict)
    aspect = "a0"

    # ---- Engine A: the MEDIA channel stages both assets under the two-layer gate
    known = _hit()
    mystery = _hit(
        media_id="web",
        page_url="https://blog.example/cliffs",
        download_url="https://pics.example/cliff.png",
        # title must share a query word or InMemoryMediaProvider filters it out
        title="Rocky mountain cliffs rising over the sea at first light",
        tags=("coast", "landscape"),
        license="unknown",  # open web declares nothing
        attribution="Mountain cliffs over the sea (https://blog.example/cliffs)",
        content=_png(marker=2),
    )
    cand = CandidateSource(
        source_id=f"{aspect}:media:morning mountain lakes",
        channel=SourceChannel.MEDIA,
        locator="morning mountain lakes",
        title="q",
        aspect_id=aspect,
        license="CC-BY-4.0",
        discovered_via="test",
    )
    source = RankedSource(
        candidate=cand,
        evaluation=SourceEvaluation(
            source_id=cand.source_id,
            shape_match=1.0,
            quality=1.0,
            license_ok=True,
            freshness_ok=True,
            estimated_yield=2,
            score=1.0,
        ),
    )
    channel = MediaChannel(provider=InMemoryMediaProvider(hits=[known, mystery]))
    ctx = FetchContext(
        job_id="job",
        store=_NullStore(),
        spec=spec,
        staging_root=tmp_path / "media_staging",
        governor=_Governor(),
        clock=lambda: datetime.now(timezone.utc),
    )

    chunks = channel.fetch(source, ctx)

    assert len(chunks) == 2
    assert {c.source_license for c in chunks} == {"CC-BY-4.0", "unknown"}
    assert all(Path(c.content_uri).exists() for c in chunks)

    # ---- Engine B: package → verify (magic-byte probe) → hf delivery
    from adh.orchestration.engine_b import run_engine_b

    result = run_engine_b(
        spec,
        work_dir=tmp_path,
        seed_engine=_ChunksAsSeeds(chunks),
        schema_base_dir=repo_root,
        resume=False,
    )

    assert result.exit_reason == "completed"
    assert result.n_records == 2
    assert result.quality_report.schema_validation_pass_rate == 1.0

    # delivered bundle: images shipped in the hf layout, provenance in the manifest
    images = sorted((result.dataset_dir / "media").glob("*.png"))
    assert len(images) == 2
    manifest_rows = [
        json.loads(line)
        for line in (result.dataset_dir / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    asset_rows = [row for row in manifest_rows if row.get("kind") == "asset"]
    assert {row["license"] for row in asset_rows} == {"CC-BY-4.0", "unknown"}
    assert all(row["attribution"] for row in asset_rows)

    # ---- evals count what distribution policy needs to know about
    schema = json.loads(
        (repo_root / "schemas" / "image_caption_record.schema.json").read_text(encoding="utf-8")
    )
    report = evaluate_dataset(result.dataset_dir, schema=schema)
    assert report.n_media_assets == 2
    assert report.unknown_license_assets == 1
    assert report.total_records == 2

    # ---- distribution is a separate consent: public refused until overridden
    refused_hub = FakeHubClient()
    with pytest.raises(PublishRefusedError) as err:
        publish_dataset(
            result.dataset_dir,
            repo_id="me/images",
            license=spec.output_license,
            private=False,
            client=refused_hub,
        )
    assert err.value.unknown_assets == 1
    assert refused_hub.calls == []  # not even repo creation happened

    private_hub = FakeHubClient()
    res_private = publish_dataset(
        result.dataset_dir,
        repo_id="me/images",
        license=spec.output_license,
        private=True,
        client=private_hub,
    )
    assert not res_private.dry_run  # private upload proceeds — allowed

    override_hub = FakeHubClient()
    res_public = publish_dataset(
        result.dataset_dir,
        repo_id="me/images",
        license=spec.output_license,
        private=False,
        client=override_hub,
        allow_unknown_licensed_assets=True,
    )
    assert not res_public.dry_run
    create = next(call for kind, call in override_hub.calls if kind == "create_repo")
    assert create["private"] is False


# ------------------------------------------------------------- CLI flag matrix
def _wire_offline_publish(monkeypatch, tmp_path):
    """Empty dataset dir + stubbed gate math: only flag threading is under test."""
    (tmp_path / "dataset").mkdir()
    report = SimpleNamespace(
        total_records=5,
        counts_by_split={"train": 5},
        duplicate_rate=0.0,
        schema_validation_pass_rate=1.0,
        contamination_residual_count=0,
        pii_residual_count=0,
        unknown_license_assets=1,
        n_media_assets=1,
        warnings=[],
    )
    monkeypatch.setattr("adh.cli.evaluate_dataset", lambda *a, **k: report)
    monkeypatch.setattr("adh.cli.assert_publishable", lambda *a, **k: None)


def test_cli_publish_threads_the_override_flag(example_spec_path, tmp_path, capsys, monkeypatch):
    captured: dict = {}

    def fake_publish(dataset_dir, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(dry_run=False, uploads=[], url="u")

    _wire_offline_publish(monkeypatch, tmp_path)
    monkeypatch.setattr("adh.cli.publish_dataset", fake_publish)

    rc = cli.main(
        [
            "publish",
            str(example_spec_path),
            "--repo-id",
            "me/ds",
            "--work-dir",
            str(tmp_path),
            "--public",
            "--allow-unknown-licenses",
        ]
    )

    assert rc == 0
    assert captured["private"] is False
    assert captured["allow_unknown_licensed_assets"] is True


def test_cli_publish_prints_the_unknown_asset_count(
    example_spec_path, tmp_path, capsys, monkeypatch
):
    _wire_offline_publish(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "adh.cli.publish_dataset",
        lambda dataset_dir, **k: SimpleNamespace(dry_run=False, uploads=[], url="u"),
    )

    rc = cli.main(
        [
            "publish",
            str(example_spec_path),
            "--repo-id",
            "me/ds",
            "--work-dir",
            str(tmp_path),
            "--public",
            "--allow-unknown-licenses",
        ]
    )

    assert rc == 0
    assert "unknown-license assets" in capsys.readouterr().out


def test_cli_publish_refusal_is_a_loud_exit_one(example_spec_path, tmp_path, capsys, monkeypatch):
    def refuse(dataset_dir, **kwargs):
        raise PublishRefusedError(4)

    _wire_offline_publish(monkeypatch, tmp_path)
    monkeypatch.setattr("adh.cli.publish_dataset", refuse)

    rc = cli.main(
        [
            "publish",
            str(example_spec_path),
            "--repo-id",
            "me/ds",
            "--work-dir",
            str(tmp_path),
            "--public",
        ]
    )

    assert rc == 1
    err = capsys.readouterr().err
    assert "unknown license" in err.lower()
    assert "--allow-unknown-licenses" in err
