"""Per-run persistence — the approval gate's durable memory (step-2 plan, T2).

Complements :mod:`adh.orchestration.state` (which owns the *acquisition* checkpoint):
a run directory holds everything needed to pause at the approval gate, survive a
restart, and resume without holding a thread:

    <work_dir>/
      run.json       <- RunRecord: status + timestamps (the state machine's node)
      spec.json      <- the submitted JobSpec
      plan.json      <- the discovered FetchPlan (persisted before the pause)
      state.json     <- acquisition checkpoint (state.py)
      manifest.jsonl / lineage.json / run_summary.json  <- engine artifacts

The module-level functions operate on one run dir (the engine uses these);
:class:`RunStore` wraps them over a base directory of run dirs for the service's
create/get/list view. File-based by design (locked decision 2) — SQLite only if
listing/filtering ever forces it.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from adh.contract import JobSpec
from adh.discovery import FetchPlan

from .state import atomic_write_text

RUN_FILE = "run.json"
SPEC_FILE = "spec.json"
PLAN_FILE = "plan.json"

# exit reasons that count as a successful terminal state (everything else FAILED)
SUCCESS_EXIT_REASONS = frozenset({"completed", "no_sources"})

RUN_ID_MAX_LENGTH = 128
RUN_ID_PATTERN = rf"^[A-Za-z0-9][A-Za-z0-9._-]{{0,{RUN_ID_MAX_LENGTH - 1}}}$"
_RUN_ID_RE = re.compile(RUN_ID_PATTERN)
_RUN_ID_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def validate_run_id(run_id: str) -> str:
    """Return a safe, single-directory run id or raise ``ValueError``.

    Keeping this check in the store protects CLI and library callers in addition
    to the HTTP validation layer.
    """
    if not isinstance(run_id, str) or _RUN_ID_RE.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must start with an ASCII letter or digit and contain at most "
            f"{RUN_ID_MAX_LENGTH} letters, digits, dots, underscores, or hyphens"
        )
    return run_id


def fresh_run_id(job_id: str) -> str:
    """Unique per-run id with a readable ``{safe(job_id)}-{suffix}`` prefix."""
    suffix = uuid.uuid4().hex[:8]
    prefix_limit = RUN_ID_MAX_LENGTH - len(suffix) - 1
    safe = _RUN_ID_UNSAFE.sub("-", job_id).strip("._-")[:prefix_limit] or "run"
    return validate_run_id(f"{safe}-{suffix}")


class RunStatus(str, Enum):
    """Nodes of the run state machine (step-2 plan §run state machine)."""

    DISCOVERING = "discovering"
    AWAITING_APPROVAL = "awaiting_approval"
    ACQUIRING = "acquiring"
    SUCCEEDED = "succeeded"
    BUILDING = "building"  # Engine B dataset build in flight over a succeeded run
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATUSES = frozenset(
    {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
)


@dataclass(frozen=True)
class RunRecord:
    """The tiny status record driving the approval-gate state machine."""

    run_id: str
    job_id: str
    status: RunStatus
    created_at: str
    updated_at: str
    exit_reason: str | None = None
    error: str | None = None

    @property
    def is_terminal(self) -> bool:
        return self.status in TERMINAL_STATUSES

    def to_dict(self) -> dict:
        return {
            "run_id": self.run_id,
            "job_id": self.job_id,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "exit_reason": self.exit_reason,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        return cls(
            run_id=data["run_id"],
            job_id=data["job_id"],
            status=RunStatus(data["status"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            exit_reason=data.get("exit_reason"),
            error=data.get("error"),
        )


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_record(run_id: str, job_id: str) -> RunRecord:
    now = utcnow_iso()
    return RunRecord(
        run_id=run_id, job_id=job_id, status=RunStatus.DISCOVERING,
        created_at=now, updated_at=now,
    )


# ------------------------------------------------------------ one run dir
def save_record(work_dir: Path | str, record: RunRecord) -> RunRecord:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    stamped = replace(record, updated_at=utcnow_iso())
    atomic_write_text(work / RUN_FILE, json.dumps(stamped.to_dict(), indent=2))
    return stamped


def load_record(work_dir: Path | str) -> RunRecord | None:
    return _load_json(Path(work_dir) / RUN_FILE, RunRecord.from_dict)


def save_spec(work_dir: Path | str, spec: JobSpec) -> None:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    atomic_write_text(work / SPEC_FILE, json.dumps(spec.model_dump(mode="json"), indent=2))


def load_spec(work_dir: Path | str) -> JobSpec | None:
    return _load_json(Path(work_dir) / SPEC_FILE, JobSpec.model_validate)


def save_plan(work_dir: Path | str, plan: FetchPlan) -> None:
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    atomic_write_text(work / PLAN_FILE, json.dumps(plan.to_dict(), indent=2))


def load_plan(work_dir: Path | str) -> FetchPlan | None:
    return _load_json(Path(work_dir) / PLAN_FILE, FetchPlan.from_dict)


def _load_json(path: Path, parse):
    """Parse ``path`` with ``parse``; a missing or corrupt file reads as absent."""
    if not path.exists():
        return None
    try:
        return parse(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, KeyError, ValueError, OSError):
        return None


# ------------------------------------------------------------ base dir of runs
class RunStore:
    """Create / read / list run dirs under one base directory (dir scan, no DB)."""

    def __init__(self, base_dir: Path | str) -> None:
        self.base_dir = Path(base_dir)

    def run_dir(self, run_id: str) -> Path:
        candidate = self.base_dir / validate_run_id(run_id)
        if not candidate.resolve().is_relative_to(self.base_dir.resolve()):
            # The lexical validation above rejects traversal. This second check
            # also refuses an existing run-directory symlink that escapes base.
            raise ValueError(f"run directory escapes configured base: {run_id!r}")
        return candidate

    def create(self, spec: JobSpec, *, run_id: str | None = None) -> RunRecord:
        job_id = spec.job_id or spec.name
        rid = run_id or fresh_run_id(job_id)
        work = self.run_dir(rid)
        save_spec(work, spec)
        return save_record(work, new_record(rid, job_id))

    def get(self, run_id: str) -> RunRecord | None:
        return load_record(self.run_dir(run_id))

    def set_status(
        self,
        run_id: str,
        status: RunStatus,
        *,
        exit_reason: str | None = None,
        error: str | None = None,
    ) -> RunRecord:
        record = self.get(run_id)
        if record is None:
            raise FileNotFoundError(f"no run record for {run_id!r} under {self.base_dir}")
        updated = replace(
            record,
            status=status,
            exit_reason=exit_reason if exit_reason is not None else record.exit_reason,
            error=error if error is not None else record.error,
        )
        return save_record(self.run_dir(run_id), updated)

    def load_spec(self, run_id: str) -> JobSpec | None:
        return load_spec(self.run_dir(run_id))

    def load_plan(self, run_id: str) -> FetchPlan | None:
        return load_plan(self.run_dir(run_id))

    def list(self) -> list[RunRecord]:
        """All runs under the base dir, newest first (by created_at).

        Runs are addressed by directory name everywhere else (``run_dir``/``get``/
        ``set_status``), so a CLI-created or renamed dir whose run.json embeds a
        different id is listed under its dir name — the addressable one.
        """
        if not self.base_dir.exists():
            return []
        records = [
            replace(record, run_id=child.name)
            for child in self.base_dir.iterdir()
            if child.is_dir() and (record := load_record(child)) is not None
        ]
        return sorted(records, key=lambda r: r.created_at, reverse=True)
