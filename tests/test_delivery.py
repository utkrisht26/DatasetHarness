"""Delivery — JSONL/Parquet shards + manifest + auto docs (Engine B 4.1), offline."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from adh.contract import Record, SourceChannel, VerifiabilityTier
from adh.delivery import datasheet_md, emit_docs, write_dataset


def _spec(output_format="jsonl"):
    from adh.contract import JobSpec

    return JobSpec(
        name="rust-qa",
        description="Q/A on rust async macros.",
        motivation="SFT data.",
        intended_uses=["supervised-fine-tuning"],
        verifiability_tier=2,
        task_categories=["question-answering"],
        languages=["en"],
        target_size={"records": 10},
        record_schema_ref="schemas/qa_record.schema.json",
        allowed_licenses=["MIT"],
        generator_models=["anthropic:claude-haiku-4-5"],
        judge_config={"models": ["anthropic:x", "openai:y"], "min_score": 3.5},
        decontaminate_against=["./eval/heldout_rust_qa.jsonl"],
        output_license="MIT",
        output_format=output_format,
        splits={"train": 0.8, "validation": 0.1, "test": 0.1},
    )


def _record(i: int, split: str) -> Record:
    return Record(
        record_id=f"rec-{i}",
        job_id="rust-qa",
        split=split,
        input=f"Question {i}?",
        output=f"Answer {i}.",
        source_channel=SourceChannel.WEB,
        source_url="https://docs.rs/tokio",
        model_id="anthropic:claude-haiku-4-5",
        prompt_hash="sha256:abc",
        generated_at=datetime.now(timezone.utc),
        pipeline_step="text-generation",
        verifiability_tier=VerifiabilityTier.SOFT,
        judge_scores={"overall": 4.0},
        judge_models=["m1", "m2"],
        dedup_cluster_id="clu-1",
    )


def _records():
    return [_record(0, "train"), _record(1, "train"), _record(2, "validation"), _record(3, "test")]


def test_write_jsonl_shards_and_manifest(tmp_path):
    result = write_dataset(_records(), spec=_spec("jsonl"), out_dir=tmp_path)
    assert result.counts_by_split == {"train": 2, "validation": 1, "test": 1}
    assert (tmp_path / "train.jsonl").exists()
    lines = (tmp_path / "train.jsonl").read_text().strip().splitlines()
    assert len(lines) == 2 and json.loads(lines[0])["job_id"] == "rust-qa"
    manifest = [json.loads(x) for x in result.manifest_path.read_text().splitlines()]
    assert manifest[0]["kind"] == "dataset" and manifest[0]["total_records"] == 4
    shard_rows = [m for m in manifest if m["kind"] == "shard"]
    assert all("sha256" in s and s["n_records"] >= 1 for s in shard_rows)


def test_write_parquet_roundtrips(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    result = write_dataset(_records(), spec=_spec("parquet"), out_dir=tmp_path)
    assert result.format == "parquet"
    table = pq.read_table(tmp_path / "train.parquet")
    assert table.num_rows == 2
    # non-scalar fields are JSON-encoded strings
    scores = json.loads(table.column("judge_scores")[0].as_py())
    assert scores == {"overall": 4.0}


def test_emit_docs_writes_datasheet_croissant_card(tmp_path):
    stats = {"total_records": 4, "counts_by_split": {"train": 2, "validation": 1, "test": 1}, "generated": 6, "accepted": 4, "accept_rate": 4 / 6}
    written = emit_docs(_spec(), stats=stats, files=["train.jsonl"], out_dir=tmp_path)
    names = {p.name for p in written}
    assert names == {"datasheet.md", "croissant.json", "README.md"}
    assert "Datasheet — rust-qa" in (tmp_path / "datasheet.md").read_text()
    croissant = json.loads((tmp_path / "croissant.json").read_text())
    assert croissant["@type"] == "Dataset" and croissant["license"] == "MIT"
    assert "task_categories:" in (tmp_path / "README.md").read_text()


def test_datasheet_reports_drop_breakdown():
    stats = {"total_records": 2, "generated": 5, "accepted": 2, "accept_rate": 0.4, "dropped_quality": 1, "dropped_jury": 2}
    md = datasheet_md(_spec(), stats)
    assert "accept rate 40.0%" in md
    assert "quality: 1" in md and "jury: 2" in md
