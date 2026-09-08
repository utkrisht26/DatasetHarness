"""Containment checks for externally supplied run identifiers."""

from __future__ import annotations

import pytest

from adh.orchestration import RunStore
from adh.orchestration.run_store import RUN_ID_MAX_LENGTH, fresh_run_id, validate_run_id


@pytest.mark.parametrize(
    "run_id",
    [
        "",
        ".",
        "..",
        "../outside",
        "nested/run",
        r"nested\run",
        "/tmp/outside",
        "has space",
    ],
)
def test_run_store_rejects_unsafe_run_ids(tmp_path, run_id: str) -> None:
    store = RunStore(tmp_path / "runs")
    with pytest.raises(ValueError, match="run_id must"):
        store.run_dir(run_id)


def test_run_store_rejects_symlink_that_escapes_base(tmp_path) -> None:
    base = tmp_path / "runs"
    outside = tmp_path / "outside"
    base.mkdir()
    outside.mkdir()
    (base / "escape").symlink_to(outside, target_is_directory=True)

    store = RunStore(base)
    with pytest.raises(ValueError, match="escapes configured base"):
        store.run_dir("escape")
    assert store.list() == []


def test_fresh_run_id_normalizes_and_bounds_untrusted_job_id() -> None:
    run_id = fresh_run_id("../" + "x" * 300)
    assert validate_run_id(run_id) == run_id
    assert len(run_id) <= RUN_ID_MAX_LENGTH
    assert not run_id.startswith(".")
