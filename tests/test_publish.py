"""Publishing + dataset-quality evaluation (Engine B 4.2/4.3), offline.

FakeHubClient stands in for ``huggingface_hub.HfApi`` (injected port); every test
is hermetic — no network, no API tokens, no optional extras required.
"""

from __future__ import annotations

import json
import sys
import types
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pytest

from adh.contract import Record, SourceChannel, VerifiabilityTier
from adh.delivery import (
    assert_publishable,
    cosine_separation,
    emit_docs,
    evaluate_dataset,
    publish_dataset,
    write_dataset,
)


def _spec():
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
        output_format="jsonl",
        splits={"train": 0.8, "validation": 0.1, "test": 0.1},
    )


_SPLITS = ("train", "train", "train", "validation", "validation", "test")


def _record(i: int, split: str, **overrides) -> Record:
    defaults = dict(
        record_id=f"rec-{i}",
        job_id="rust-qa",
        split=split,
        input=f"Q{i}: " + "word " * (i + 2),
        output=f"A{i}.",
        source_channel=SourceChannel.WEB,
        source_url=f"https://docs.rs/tokio/{i}",
        source_license="MIT",
        model_id="anthropic:claude-haiku-4-5",
        prompt_hash="sha256:abc",
        generated_at=datetime.now(timezone.utc),
        pipeline_step="text-generation",
        verifiability_tier=VerifiabilityTier.SOFT,
        judge_scores={"overall": 4.0},
        judge_models=["m1", "m2"],
        dedup_cluster_id=f"clu-{i}",
    )
    defaults.update(overrides)
    return Record(**defaults)


def _records() -> list[Record]:
    """7 records: one payload-duplicate pair, one contaminated, one unknown-license."""
    recs = [_record(i, split) for i, split in enumerate(_SPLITS)]
    dup = _record(99, "train", input=recs[0].input, output=recs[0].output)
    recs[3] = _record(3, "validation", contamination_flag=True)
    recs[4] = _record(4, "validation", source_license="unknown")
    recs[5] = _record(5, "test", source_channel=SourceChannel.SYNTHESIS, source_url=None)
    return [*recs, dup]


def _clean_records() -> list[Record]:
    """6 records with no duplicates, contamination, or unknown licenses."""
    return [_record(i, split) for i, split in enumerate(_SPLITS)]


def _write_dataset_with(records: list[Record], tmp_path: Path, name: str = "dataset") -> Path:
    out = tmp_path / name
    spec = _spec()
    result = write_dataset(records, spec=spec, out_dir=out)
    stats = {
        "total_records": result.total_records,
        "counts_by_split": result.counts_by_split,
        "generated": 8,
        "accepted": result.total_records,
        "accept_rate": 0.75,
    }
    emit_docs(spec, stats=stats, files=[p.name for p in result.files], out_dir=out)
    return out


def _dataset_dir(tmp_path: Path) -> Path:
    return _write_dataset_with(_records(), tmp_path)


def _clean_dataset_dir(tmp_path: Path) -> Path:
    return _write_dataset_with(_clean_records(), tmp_path, name="clean")


@dataclass
class FakeHubClient:
    """Records every call; returns deterministic fake URLs."""

    calls: list[tuple[str, dict]] = field(default_factory=list)

    def create_repo(
        self,
        repo_id: str,
        *,
        repo_type: str = "dataset",
        private: bool = True,
        exist_ok: bool = True,
    ) -> str:
        self.calls.append(
            ("create_repo", {"repo_id": repo_id, "repo_type": repo_type, "private": private})
        )
        return f"https://huggingface.co/{repo_type}s/{repo_id}"

    def upload_file(
        self,
        *,
        path_or_fileobj: str,
        path_in_repo: str,
        repo_id: str,
        repo_type: str = "dataset",
        commit_message: str = "",
    ) -> str:
        self.calls.append(
            (
                "upload_file",
                {
                    "path": str(path_or_fileobj),
                    "path_in_repo": path_in_repo,
                    "repo_id": repo_id,
                    "commit_message": commit_message,
                },
            )
        )
        return f"https://huggingface.co/{repo_type}s/{repo_id}/blob/main/{path_in_repo}"


@dataclass
class FakeRedactor:
    """Redacts a canned token; residual = record whose text would change."""

    marker: str = "555-0100"

    def redact(self, text: str) -> str:
        return text.replace(self.marker, "[REDACTED]")


# --- publish_dataset ----------------------------------------------------------


def test_publish_uploads_full_plan_and_returns_urls(tmp_path):
    d = _dataset_dir(tmp_path)
    fake = FakeHubClient()
    result = publish_dataset(
        d, repo_id="acme/rust-qa", license="CC-BY-4.0", private=True, client=fake
    )

    creates = [c for c in fake.calls if c[0] == "create_repo"]
    uploads = [c for c in fake.calls if c[0] == "upload_file"]
    assert len(creates) == 1
    assert creates[0][1] == {"repo_id": "acme/rust-qa", "repo_type": "dataset", "private": True}

    remotes = [c[1]["path_in_repo"] for c in uploads]
    assert set(remotes) == {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "manifest.jsonl",
        "datasheet.md",
        "croissant.json",
        "README.md",
    }
    assert remotes[-1] == "README.md"  # card goes last
    assert result.dry_run is False
    assert result.private is True
    assert result.url == "https://huggingface.co/datasets/acme/rust-qa"
    assert result.uploads[-1].url.endswith("/blob/main/README.md")


def test_publish_accepts_delivery_result(tmp_path):
    spec = _spec()
    result = write_dataset(_records(), spec=spec, out_dir=tmp_path / "ds")
    fake = FakeHubClient()
    published = publish_dataset(result, repo_id="acme/x", license="MIT", client=fake)
    remotes = {u.repo_path for u in published.uploads}
    assert {"train.jsonl", "validation.jsonl", "manifest.jsonl", "README.md"} <= remotes


def test_publish_stamps_license_and_provenance_into_card(tmp_path):
    d = _dataset_dir(tmp_path)
    fake = FakeHubClient()
    publish_dataset(
        d,
        repo_id="acme/rust-qa",
        license="CC-BY-4.0",
        task_categories=["question-answering"],
        languages=["en"],
        client=fake,
    )

    card = (d / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---")
    assert "license: cc-by-4.0" in card  # overrides whatever the build stamped
    assert "- question-answering" in card
    assert "- en" in card
    assert "## Source provenance" in card
    assert "web" in card and "synthesis" in card
    assert "docs.rs" in card

    uploaded_card = next(
        c for c in fake.calls if c[0] == "upload_file" and c[1]["path_in_repo"] == "README.md"
    )
    assert Path(uploaded_card[1]["path"]).exists()


def test_dry_run_makes_zero_calls_and_zero_writes(tmp_path):
    d = _dataset_dir(tmp_path)
    before = {p.name: p.read_bytes() for p in d.iterdir()}
    fake = FakeHubClient()

    result = publish_dataset(d, repo_id="acme/rust-qa", license="MIT", client=fake, dry_run=True)

    assert fake.calls == []
    assert result.dry_run is True
    assert result.url is None
    assert {u.repo_path for u in result.uploads} == {
        "train.jsonl",
        "validation.jsonl",
        "test.jsonl",
        "manifest.jsonl",
        "datasheet.md",
        "croissant.json",
        "README.md",
    }
    assert all(u.url is None for u in result.uploads)
    after = {p.name: p.read_bytes() for p in d.iterdir()}
    assert before == after  # no card rewrite, no side effects


def test_lazy_import_failure_mentions_extra(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)  # forces ImportError
    d = _dataset_dir(tmp_path)
    with pytest.raises(RuntimeError, match=r"adh\[publish\]"):
        publish_dataset(d, repo_id="acme/rust-qa", license="MIT")


def test_default_client_routes_through_hfapi_adapter(monkeypatch, tmp_path):
    recorded: list[tuple[str, dict]] = []

    class StubHfApi:
        def __init__(self):
            pass

        def create_repo(self, repo_id, *, repo_type, private, exist_ok):
            recorded.append(("create_repo", {"repo_id": repo_id, "private": private}))
            return f"https://huggingface.co/{repo_type}s/{repo_id}"

        def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
            recorded.append(("upload_file", {"path_in_repo": path_in_repo}))
            commit = f"https://huggingface.co/datasets/{repo_id}/commit/deadbeef"
            return types.SimpleNamespace(commit_url=commit)

    stub = types.ModuleType("huggingface_hub")
    stub.HfApi = StubHfApi  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", stub)

    d = _dataset_dir(tmp_path)
    result = publish_dataset(d, repo_id="acme/rust-qa", license="MIT")
    assert ("create_repo", {"repo_id": "acme/rust-qa", "private": True}) in recorded
    assert result.uploads[0].url.endswith("/commit/deadbeef")


def test_publish_can_clear_optional_card_metadata(tmp_path):
    d = _dataset_dir(tmp_path)
    (d / "README.md").write_text(
        "---\n"
        "license: old\n"
        "task_categories: [old-task]\n"
        "language: [fr]\n"
        "custom: retained\n"
        "---\n\n"
        "# Existing body\n",
        encoding="utf-8",
    )

    publish_dataset(
        d,
        repo_id="acme/rust-qa",
        license=" MIT ",
        task_categories=[],
        languages=[],
        client=FakeHubClient(),
    )

    card = (d / "README.md").read_text(encoding="utf-8")
    assert "license: mit" in card
    assert "custom: retained" in card
    assert "task_categories:" not in card
    assert "language:" not in card


def test_publish_rejects_manifest_paths_outside_dataset(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    outside = tmp_path / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    (d / "manifest.jsonl").write_text(
        json.dumps({"kind": "shard", "path": "../outside.jsonl"}) + "\n",
        encoding="utf-8",
    )
    fake = FakeHubClient()

    with pytest.raises(ValueError, match="outside"):
        publish_dataset(d, repo_id="acme/rust-qa", license="MIT", client=fake)

    assert fake.calls == []
    assert not (d / "README.md").exists()


def test_publish_reports_malformed_manifest_line(tmp_path):
    d = tmp_path / "dataset"
    d.mkdir()
    (d / "manifest.jsonl").write_text(
        '{"kind": "dataset", "job_id": "example"}\nnot-json\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"manifest\.jsonl at line 2"):
        publish_dataset(d, repo_id="acme/rust-qa", license="MIT", dry_run=True)


# --- evaluate_dataset ---------------------------------------------------------


def test_evaluate_metrics_math_on_synthetic_dataset(tmp_path):
    d = _dataset_dir(tmp_path)
    report = evaluate_dataset(d)

    assert report.total_records == 7
    assert report.counts_by_split == {"train": 4, "validation": 2, "test": 1}
    assert report.duplicate_records == 1
    assert report.duplicate_rate == pytest.approx(1 / 7)
    assert report.contamination_residual_count == 1
    assert report.unknown_license_records == 1
    assert report.license_coverage == pytest.approx(6 / 7)
    assert report.schema_validation_pass_rate == pytest.approx(1.0)
    assert report.pii_residual_count == 0
    assert report.warnings == ()

    lens = sorted(len(r.input) for r in _records())
    assert report.input_lengths.minimum == lens[0]
    assert report.input_lengths.maximum == max(lens)
    assert report.input_lengths.n == 7
    assert report.input_lengths.mean == pytest.approx(sum(lens) / 7)
    assert report.input_lengths.p50 <= report.input_lengths.maximum
    assert report.split_fractions["train"] == pytest.approx(4 / 7)


def test_evaluate_schema_pass_rate_with_explicit_schema(tmp_path):
    d = tmp_path / "raw"
    d.mkdir()
    rows = [
        {"record_id": "r0", "input": "q0", "output": "a0"},
        {"record_id": "r-broken", "input": "q"},  # missing output
        {"record_id": "r2", "input": "q2", "output": "a2"},
        {"record_id": "r3", "input": "q3", "output": "a3"},
    ]
    (d / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    schema = {"type": "object", "required": ["record_id", "input", "output"]}
    report = evaluate_dataset(d, schema=schema)
    assert report.total_records == 4
    assert report.schema_validation_pass_rate == pytest.approx(0.75)
    assert len(report.schema_failures) == 1
    assert "output" in report.schema_failures[0]


def test_evaluate_pii_residuals_via_injected_redactor(tmp_path):
    d = tmp_path / "pii"
    d.mkdir()
    rows = [
        {"record_id": "a", "input": "call 555-0100", "output": "ok"},
        {"record_id": "b", "input": "clean", "output": "fine"},
        {"record_id": "c", "output": "id 555-0100 please"},
    ]
    (d / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    assert evaluate_dataset(d, pii_redactor=FakeRedactor()).pii_residual_count == 2
    assert evaluate_dataset(d).pii_residual_count == 0


def test_evaluate_references_surface_contamination_residuals(tmp_path):
    heldout = " ".join(["token"] * 13)
    ref = tmp_path / "heldout.jsonl"
    ref.write_text(json.dumps({"question": heldout}) + "\n", encoding="utf-8")

    d = tmp_path / "ref"
    d.mkdir()
    rows = [
        {"record_id": "dirty", "input": "prefix", "output": f"leaked {heldout} suffix"},
        {"record_id": "clean", "input": "unrelated content here", "output": "fine"},
    ]
    (d / "train.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

    report = evaluate_dataset(d, references=[str(ref)])
    assert report.contamination_residual_count == 1
    # flags alone would have missed it — the reference scan caught the leak
    assert evaluate_dataset(d).contamination_residual_count == 0


def test_evaluate_falls_back_to_glob_without_manifest(tmp_path):
    d = tmp_path / "loose"
    d.mkdir()
    (d / "train.jsonl").write_text('{"record_id":"x","input":"i","output":"o"}\n', encoding="utf-8")
    report = evaluate_dataset(d)
    assert report.total_records == 1
    assert any("manifest" in w for w in report.warnings)


# --- assert_publishable gate --------------------------------------------------


def test_assert_publishable_passes_clean_report(tmp_path):
    report = evaluate_dataset(_clean_dataset_dir(tmp_path))
    assert_publishable(report, min_records=6)


def test_assert_publishable_collects_reasons(tmp_path):
    report = evaluate_dataset(_dataset_dir(tmp_path))
    with pytest.raises(ValueError) as excinfo:
        assert_publishable(report, min_records=100, max_contamination=0)
    reasons = getattr(excinfo.value, "reasons")
    assert any("records" in r for r in reasons)
    assert any("contamination" in r for r in reasons)

    broken = evaluate_dataset(
        _dataset_dir(tmp_path),
        schema={"type": "object", "required": ["nonexistent_field"]},
    )
    with pytest.raises(ValueError, match="schema"):
        assert_publishable(broken, min_records=1)


def test_report_renders_markdown_table(tmp_path):
    md = evaluate_dataset(_dataset_dir(tmp_path)).to_markdown()
    assert "| Metric | Value |" in md
    assert "total_records" in md and "| 7 |" in md
    assert "split_balance" in md and "train: 4 (57.1%)" in md
    assert "duplicate_rate" in md
    assert "schema_validation_pass_rate" in md
    assert "contamination_residuals" in md


# --- downstream-lift probe ----------------------------------------------------


def test_cosine_separation_clusters_vs_random_pairs():
    groups = {
        "aspect_a": [[1.0, 0.0], [0.98, 0.19], [1.0, 0.05]],
        "aspect_b": [[0.0, 1.0], [0.19, 0.98], [0.05, 1.0]],
    }
    separation = cosine_separation(groups, pairs_per_group=10, seed=0)
    assert separation > 0.8  # within-aspect pairs ~cos 1, cross pairs ~cos 0


def test_cosine_separation_degenerate_input_is_zero():
    assert cosine_separation({"only": [[1.0, 0.0], [0.0, 1.0]]}) == 0.0
    assert cosine_separation({}) == 0.0


def test_embedding_lift_probe_requires_optional_extra(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    from adh.delivery import EmbeddingLiftProbe

    probe = EmbeddingLiftProbe(model_name="fake/model")
    with pytest.raises(RuntimeError, match=r"adh\[evals\]"):
        probe.measure({"a": ["one two", "three four"], "b": ["five six", "seven eight"]})


# ------------------------------------------------- unknown-license publish gate
def _append_asset_rows(dataset_dir: Path, licenses: list[str]) -> Path:
    """Simulate delivered media: manifest ``kind:"asset"`` rows carrying license."""
    with (dataset_dir / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        for i, lic in enumerate(licenses):
            fh.write(
                json.dumps(
                    {
                        "kind": "asset",
                        "split": "train",
                        "path": f"media/asset{i}.png",
                        "format": "png",
                        "sha256": "a" * 64,
                        "bytes": 128,
                        "duration_s": None,
                        "width": 320,
                        "height": 240,
                        "license": lic,
                        "attribution": "someone (https://example.com/i)",
                        "record_ids": [f"r-{i}"],
                    }
                )
                + "\n"
            )
    return dataset_dir


def test_public_upload_refused_while_unknown_assets_present(tmp_path):
    from adh.delivery.publish import PublishRefusedError, publish_dataset

    d = _append_asset_rows(_clean_dataset_dir(tmp_path), ["CC-BY-4.0", "unknown"])
    hub = FakeHubClient()

    with pytest.raises(PublishRefusedError) as err:
        publish_dataset(d, repo_id="me/ds", license="CC-BY-4.0", private=False, client=hub)

    assert err.value.unknown_assets == 1
    assert hub.calls == []  # nothing at all reached the Hub


def test_private_upload_ignores_unknown_assets(tmp_path):
    from adh.delivery.publish import publish_dataset

    d = _append_asset_rows(_clean_dataset_dir(tmp_path), ["unknown"])
    hub = FakeHubClient()

    result = publish_dataset(d, repo_id="me/ds", license="CC-BY-4.0", private=True, client=hub)

    assert not result.dry_run
    create = next(call for kind, call in hub.calls if kind == "create_repo")
    assert create["private"] is True


def test_explicit_override_publishes_public_with_unknown_assets(tmp_path):
    from adh.delivery.publish import publish_dataset

    d = _append_asset_rows(_clean_dataset_dir(tmp_path), ["unknown"])
    hub = FakeHubClient()

    result = publish_dataset(
        d,
        repo_id="me/ds",
        license="CC-BY-4.0",
        private=False,
        client=hub,
        allow_unknown_licensed_assets=True,
    )

    assert not result.dry_run
    create = next(call for kind, call in hub.calls if kind == "create_repo")
    assert create["private"] is False


def test_dry_run_never_refuses_unknown_assets(tmp_path):
    from adh.delivery.publish import publish_dataset

    d = _append_asset_rows(_clean_dataset_dir(tmp_path), ["unknown"])
    hub = FakeHubClient()

    result = publish_dataset(
        d, repo_id="me/ds", license="CC-BY-4.0", private=False, client=hub, dry_run=True
    )

    assert result.dry_run
    assert hub.calls == []


def test_text_records_with_unknown_licenses_do_not_block_publication(tmp_path):
    """The distribution gate counts ASSET rows; text keeps its advisory role."""
    from adh.delivery.publish import publish_dataset

    d = _write_dataset_with(
        _records(), tmp_path, name="texty"
    )  # has an unknown-license TEXT record
    hub = FakeHubClient()

    result = publish_dataset(d, repo_id="me/ds", license="CC-BY-4.0", private=False, client=hub)

    assert not result.dry_run


def test_evals_counts_unknown_license_assets(tmp_path):
    from adh.delivery.evals import evaluate_dataset

    d = _append_asset_rows(_clean_dataset_dir(tmp_path), ["MIT", "unknown", "unknown"])

    report = evaluate_dataset(d)

    assert report.n_media_assets == 3
    assert report.unknown_license_assets == 2
    assert "unknown_license_assets" in report.to_markdown()
