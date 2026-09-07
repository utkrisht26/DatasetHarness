"""Durable execution state — resume cleanly after a kill (OUTLINE §1.5).

A harness-level checkpoint: after each source is fetched, the set of completed source
ids + accumulated metrics are persisted to ``state.json``. A re-run skips completed
sources and reloads prior chunks from the manifest, so work is never repeated. (When
the deepagents/LangGraph backend is used, its ``checkpointer`` provides the same
guarantee at the graph level; this is the equivalent for the offline pipeline.)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

from .limits import RunMetrics

logger = logging.getLogger(__name__)


def atomic_write_text(path: Path, text: str) -> None:
    """Write via a temp file + rename so a concurrent reader never sees a torn file.

    The service reads these files from request threads while engine workers write
    them; ``Path.write_text`` truncates in place and can expose partial JSON.
    """
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


@dataclass
class JobState:
    job_id: str
    completed_source_ids: set[str] = field(default_factory=set)
    metrics: RunMetrics = field(default_factory=RunMetrics)
    exit_reason: str | None = None
    notes: list[str] = field(default_factory=list)

    def mark_done(self, source_id: str) -> None:
        self.completed_source_ids.add(source_id)

    def is_done(self, source_id: str) -> bool:
        return source_id in self.completed_source_ids

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "completed_source_ids": sorted(self.completed_source_ids),
            "metrics": self.metrics.to_dict(),
            "exit_reason": self.exit_reason,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "JobState":
        return cls(
            job_id=data["job_id"],
            completed_source_ids=set(data.get("completed_source_ids", [])),
            metrics=RunMetrics.from_dict(data.get("metrics", {})),
            exit_reason=data.get("exit_reason"),
            notes=list(data.get("notes", [])),
        )


def state_path(work_dir: Path) -> Path:
    return Path(work_dir) / "state.json"


def load_state(work_dir: Path, job_id: str) -> JobState:
    """Load prior state for ``job_id`` if present and matching; else a fresh state.

    Read paths must survive a torn file, so corruption never raises — but the
    lost checkpoint (completed sources + breaker metrics) is announced loudly:
    silently resetting it would quietly lift the run's spend ceilings.
    """
    p = state_path(work_dir)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if data.get("job_id") == job_id:
                return JobState.from_dict(data)
            logger.warning(
                "%s holds state for job %r, expected %r — starting fresh",
                p,
                data.get("job_id"),
                job_id,
            )
            return JobState(job_id=job_id)
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "%s is unreadable (%s) — starting fresh; the prior checkpoint "
                "(completed sources, metrics) is lost",
                p,
                exc,
            )
    return JobState(job_id=job_id)


def save_state(work_dir: Path, state: JobState) -> Path:
    p = state_path(work_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, json.dumps(state.to_dict(), indent=2))
    return p
