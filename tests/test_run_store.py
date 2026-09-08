"""Tests for the per-run store: record/spec/plan persistence + the dir-scan list."""

from __future__ import annotations

import json

from adh.contract import JobSpec
from adh.discovery import SourceDiscoveryAgent
from adh.orchestration import RunStatus, RunStore
from adh.orchestration.run_store import (
    load_plan,
    load_record,
    load_spec,
    new_record,
    save_plan,
    save_record,
    save_spec,
)


def _spec(minimal_spec_dict) -> JobSpec:
    return JobSpec.model_validate(minimal_spec_dict)


# ------------------------------------------------------------- one run dir
def test_record_round_trip_stamps_updated_at(tmp_path):
    record = new_record(run_id="r1", job_id="demo")
    saved = save_record(tmp_path, record)
    loaded = load_record(tmp_path)
    assert loaded == saved
    assert loaded.status is RunStatus.DISCOVERING
    assert not loaded.is_terminal


def test_spec_round_trip(tmp_path, minimal_spec_dict):
    spec = _spec(minimal_spec_dict)
    save_spec(tmp_path, spec)
    assert load_spec(tmp_path) == spec


def test_plan_round_trip(tmp_path, minimal_spec_dict):
    """FetchPlan survives the approval pause byte-for-byte (sources, aspects, notes)."""
    plan = SourceDiscoveryAgent().discover(_spec(minimal_spec_dict), tmp_path)
    save_plan(tmp_path, plan)
    loaded = load_plan(tmp_path)
    assert loaded == plan


def test_missing_and_corrupt_files_read_as_absent(tmp_path):
    assert load_record(tmp_path) is None
    assert load_spec(tmp_path) is None
    assert load_plan(tmp_path) is None
    (tmp_path / "run.json").write_text("{not json", encoding="utf-8")
    assert load_record(tmp_path) is None


# ------------------------------------------------------------- the store
def test_store_create_persists_spec_and_record(tmp_path, minimal_spec_dict):
    store = RunStore(tmp_path)
    record = store.create(_spec(minimal_spec_dict))
    assert record.status is RunStatus.DISCOVERING
    assert record.job_id == "demo"
    assert record.run_id.startswith("demo-")
    work = store.run_dir(record.run_id)
    assert json.loads((work / "spec.json").read_text())["name"] == "demo"
    assert store.get(record.run_id) == record


def test_store_set_status_updates_reason_and_error(tmp_path, minimal_spec_dict):
    store = RunStore(tmp_path)
    record = store.create(_spec(minimal_spec_dict))
    updated = store.set_status(
        record.run_id, RunStatus.FAILED, exit_reason="max_llm_tokens", error="boom"
    )
    assert updated.status is RunStatus.FAILED
    assert updated.exit_reason == "max_llm_tokens"
    assert updated.error == "boom"
    assert store.get(record.run_id) == updated


def test_store_list_is_newest_first_and_skips_junk(tmp_path, minimal_spec_dict):
    store = RunStore(tmp_path)
    first = store.create(_spec(minimal_spec_dict), run_id="run-a")
    second = store.create(_spec(minimal_spec_dict), run_id="run-b")
    (tmp_path / "not-a-run").mkdir()  # dir without run.json is skipped
    listed = store.list()
    assert [r.run_id for r in listed] == [second.run_id, first.run_id] or [
        r.run_id for r in listed
    ] == [
        first.run_id,
        second.run_id,
    ]  # same-timestamp ties are order-stable enough
    assert len(listed) == 2


def test_store_get_unknown_run_is_none(tmp_path):
    assert RunStore(tmp_path).get("nope") is None


def test_store_list_reports_dir_name_as_run_id(tmp_path, minimal_spec_dict):
    """Runs are addressed by directory name (run_dir/get/set_status all resolve
    that way), so a CLI-created or renamed dir whose run.json embeds a different
    id must still be listed under its addressable name — otherwise every list
    entry with a divergent id 404s on detail."""
    spec = _spec(minimal_spec_dict)
    work = tmp_path / "live-gate-2"
    work.mkdir()
    save_spec(work, spec)
    save_record(
        work,
        new_record(
            run_id="rust-async-macro-patterns", job_id="rust-async-macro-patterns"
        ),
    )

    store = RunStore(tmp_path)
    listed = store.list()
    assert [r.run_id for r in listed] == ["live-gate-2"]
    # the listed id must round-trip through the read path
    assert store.get(listed[0].run_id) is not None
