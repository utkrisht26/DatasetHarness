"""``adh publish`` — quality-gated Hub upload over an Engine B dataset dir.

The command must (a) refuse to publish anything that fails
:data:`adh.delivery.assert_publishable`, (b) stay private by default,
(c) stamp + upload the dataset card last, and (d) plan uploads without any
network access under ``--dry-run``. Every test here runs fully offline via an
injected :class:`HubClient` double.
"""

from __future__ import annotations

import copy
import dataclasses
import shutil
from pathlib import Path

from adh.channels import DatasetsChannel, InMemoryDatasetLoader
from adh.contract import JobSpec, SourceChannel
from adh.delivery import DatasetQualityReport, evaluate_dataset
from adh.discovery import AutoApprover, SourceDiscoveryAgent
from adh.generation import HeuristicGenerator
from adh.orchestration import run_engine_a, run_engine_b
from adh.search import DatasetHit, InMemoryDatasetRegistry, InMemorySearchProvider
from adh.verification import HeuristicJudge

RUST_DS = DatasetHit(
    dataset_id="org/rust-async-macros-qa",
    locator="https://huggingface.co/datasets/org/rust-async-macros-qa",
    title="Rust async macro Q/A",
    description="rust async macro design declarative procedural question-answering input output",
    license="MIT",
    downloads=5000,
    last_modified="2024-09-01",
)


class _FakeHub:
    """HubClient double that records every call and never touches a network."""

    def __init__(self):
        self.created: list[tuple[str, bool]] = []
        self.uploaded: list[str] = []

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):  # noqa: ARG002
        self.created.append((repo_id, private))
        return f"https://huggingface.co/{repo_type}s/{repo_id}"

    def upload_file(self, *, path_or_fileobj, path_in_repo, repo_id, repo_type, commit_message):
        del repo_id, repo_type, commit_message  # recorded via uploaded paths only
        assert Path(path_or_fileobj).exists(), f"uploaded a missing file: {path_or_fileobj}"
        self.uploaded.append(path_in_repo)
        return f"https://huggingface.co/datasets/acme/rust-qa/blob/main/{path_in_repo}"


class _ExplosiveHub(_FakeHub):
    """A client whose every call fails — proves dry runs never construct/use one."""

    def create_repo(self, *a, **kw):
        raise AssertionError("dry run created a repo")

    def upload_file(self, *a, **kw):
        raise AssertionError("dry run uploaded a file")


def _spec_dict(minimal_spec_dict) -> dict:
    d = copy.deepcopy(minimal_spec_dict)
    d["description"] = "Rust async macro patterns. Tokio runtime internals scheduler."
    d["motivation"] = "teach async"
    d["target_size"] = {"records": 12}
    d["generator_models"] = []
    return d


def _prepare_run(minimal_spec_dict, repo_root, tmp_path) -> tuple[JobSpec, object]:
    """Engine A -> Engine B fully offline; returns (spec_path, work_dir)."""
    import yaml

    spec_dict = _spec_dict(minimal_spec_dict)
    spec_path = tmp_path / "specs" / "pub-spec.yaml"
    spec_path.parent.mkdir()
    spec_path.write_text(yaml.safe_dump(spec_dict), encoding="utf-8")
    # record_schema_ref / decontaminate_against resolve against the spec base dir
    (tmp_path / "schemas").mkdir()
    shutil.copy(repo_root / "schemas" / "qa_record.schema.json", tmp_path / "schemas")
    (tmp_path / "eval").mkdir()
    shutil.copy(repo_root / "eval" / "heldout_rust_qa.jsonl", tmp_path / "eval")

    spec = JobSpec.model_validate(spec_dict)
    work = tmp_path / "work"
    agent = SourceDiscoveryAgent(
        registry=InMemoryDatasetRegistry(
            datasets=[RUST_DS],
            samples={RUST_DS.dataset_id: "rust async macro declarative procedural input output"},
        ),
        search=InMemorySearchProvider(),
    )
    channels = {
        SourceChannel.DATASETS: DatasetsChannel(
            loader=InMemoryDatasetLoader(
                records={"org/rust-async-macros-qa": ["Q: macro? A: macro_rules!"]}
            )
        )
    }
    ea = run_engine_a(spec, work_dir=work, agent=agent, channels=channels,
                      approver=AutoApprover())
    assert ea.exit_reason == "completed"
    eb = run_engine_b(spec, work_dir=work, generator=HeuristicGenerator(),
                      judge=HeuristicJudge(), schema_base_dir=tmp_path)
    assert eb.n_records > 0
    return spec_path, work


def test_publish_uploads_card_last_and_is_private_by_default(
    minimal_spec_dict, repo_root, tmp_path, capsys, monkeypatch
):
    from adh import cli

    spec_path, work = _prepare_run(minimal_spec_dict, repo_root, tmp_path)
    hub = _FakeHub()
    monkeypatch.setattr(cli, "default_hub_client", lambda: hub)

    rc = cli.main([
        "publish", str(spec_path), "--repo-id", "acme/rust-qa",
        "--work-dir", str(work),
    ])

    assert rc == 0
    assert hub.created == [("acme/rust-qa", True)]  # private is the default
    assert hub.uploaded[-1] == "README.md"  # stamped card goes last
    assert any(p.endswith(".jsonl") for p in hub.uploaded)  # shards went up too
    out = capsys.readouterr().out
    assert "acme/rust-qa" in out
    assert "private" in out.lower()


def test_publish_dry_run_plans_without_touching_the_hub(
    minimal_spec_dict, repo_root, tmp_path, capsys, monkeypatch
):
    from adh import cli

    spec_path, work = _prepare_run(minimal_spec_dict, repo_root, tmp_path)
    hub = _ExplosiveHub()
    monkeypatch.setattr(cli, "default_hub_client", lambda: hub)

    rc = cli.main([
        "publish", str(spec_path), "--repo-id", "acme/rust-qa",
        "--work-dir", str(work), "--dry-run",
    ])

    assert rc == 0
    out = capsys.readouterr().out
    assert "dry" in out.lower()
    assert ".jsonl" in out  # the planned upload list names the shards


def test_publish_refuses_when_the_quality_gate_fails(
    minimal_spec_dict, repo_root, tmp_path, capsys, monkeypatch
):
    from adh import cli

    spec_path, work = _prepare_run(minimal_spec_dict, repo_root, tmp_path)
    hub = _FakeHub()
    monkeypatch.setattr(cli, "default_hub_client", lambda: hub)

    real_evaluate = evaluate_dataset

    def contaminated(dataset_dir, **kw):
        report = real_evaluate(dataset_dir, **kw)
        return dataclasses.replace(report, contamination_residual_count=7)

    monkeypatch.setattr(cli, "evaluate_dataset", contaminated)

    rc = cli.main([
        "publish", str(spec_path), "--repo-id", "acme/rust-qa",
        "--work-dir", str(work),
    ])

    assert rc != 0
    assert hub.created == [] and hub.uploaded == []  # nothing left the machine
    err = capsys.readouterr().err
    assert "contamination" in err.lower()
    assert "refus" in err.lower()


def test_publish_requires_a_built_dataset(minimal_spec_dict, repo_root, tmp_path, capsys):
    import yaml

    from adh import cli

    spec_path = tmp_path / "specs" / "pub-spec.yaml"
    spec_path.parent.mkdir()
    spec_path.write_text(yaml.safe_dump(_spec_dict(minimal_spec_dict)), encoding="utf-8")

    rc = cli.main([
        "publish", str(spec_path), "--repo-id", "acme/rust-qa",
        "--work-dir", str(tmp_path / "nowhere"),
    ])

    assert rc == 1
    assert "build" in capsys.readouterr().err.lower()


def test_publish_missing_hub_extra_prints_install_hint_not_a_traceback(
    minimal_spec_dict, repo_root, tmp_path, capsys, monkeypatch
):
    from adh import cli

    spec_path, work = _prepare_run(minimal_spec_dict, repo_root, tmp_path)

    def no_hub():
        raise RuntimeError(
            "publishing to the HF Hub requires 'huggingface_hub'; "
            "install with: pip install 'adh[publish]'"
        )

    monkeypatch.setattr(cli, "default_hub_client", no_hub)

    rc = cli.main([
        "publish", str(spec_path), "--repo-id", "acme/rust-qa",
        "--work-dir", str(work),
    ])

    assert rc == 1
    err = capsys.readouterr().err
    assert "adh[publish]" in err
    assert "Traceback" not in err


def test_quality_report_markdown_renders_split_balance(tmp_path):
    """Sanity on the card the gate prints: the metric table renders counts."""
    from pathlib import Path

    from adh.delivery.evals import LengthStats

    stats = LengthStats(n=2, minimum=1, mean=2.0, p50=2.0, p95=3.0, maximum=3)
    report = DatasetQualityReport(
        dataset_dir=Path(tmp_path), total_records=10,
        counts_by_split={"train": 9, "test": 1},
        split_fractions={"train": 0.9, "test": 0.1},
        input_lengths=stats, output_lengths=stats,
        duplicate_records=1, duplicate_rate=0.1,
        contamination_residual_count=0, license_coverage=1.0,
        unknown_license_records=0, schema_validation_pass_rate=1.0,
        schema_failures=(), pii_residual_count=0, warnings=(),
    )
    md = report.to_markdown()
    assert "| total_records | 10 |" in md
    assert "train" in md
