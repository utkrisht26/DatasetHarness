"""Durable, structured execution telemetry for one harness run.

Events are appended as JSON Lines inside the run directory.  The format is
deliberately small and dependency-free so telemetry remains available while
providers, callbacks, or the run itself are failing.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

TELEMETRY_FILE = "telemetry.jsonl"
TELEMETRY_SCHEMA_VERSION = 1
MAX_ATTRIBUTE_STRING = 2_000
MAX_ATTRIBUTE_ITEMS = 50

_SENSITIVE_KEY = re.compile(
    r"(?:api[-_]?key|authorization|cookie|credential|password|secret|token)$",
    re.IGNORECASE,
)
_BEARER_VALUE = re.compile(r"\bBearer\s+\S+", re.IGNORECASE)
_COMMON_KEY_VALUE = re.compile(r"\b(?:sk|sk-proj|sk-or-v1)-[A-Za-z0-9_-]{12,}\b")
_JSON_SECRET_VALUE = re.compile(
    r'("(?:api[-_]?key|authorization|cookie|credential|password|secret|(?:access_|refresh_)?token)"'
    r'\s*:\s*)"(?:\\.|[^"\\])*"',
    re.IGNORECASE,
)


@dataclass
class _FileState:
    lock: threading.Lock
    sequence: int


_states_lock = threading.Lock()
_states: dict[Path, _FileState] = {}
_current_recorder: ContextVar[TelemetryRecorder | None] = ContextVar(
    "adh_telemetry_recorder", default=None
)


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _existing_event_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("rb") as stream:
            return sum(1 for line in stream if line.strip())
    except OSError:
        return 0


def _state_for(path: Path) -> _FileState:
    resolved = path.resolve()
    with _states_lock:
        state = _states.get(resolved)
        if state is None:
            state = _FileState(lock=threading.Lock(), sequence=_existing_event_count(resolved))
            _states[resolved] = state
        return state


def _redact_string(value: str, *, truncate: bool = True) -> str:
    # Tool-call arguments are themselves JSON strings. Redact credential values
    # there without reserializing the string we need to inspect for parse errors.
    redacted = _JSON_SECRET_VALUE.sub(r'\1"[REDACTED]"', value)
    redacted = _BEARER_VALUE.sub("Bearer [REDACTED]", redacted)
    redacted = _COMMON_KEY_VALUE.sub("[REDACTED]", redacted)
    if truncate and len(redacted) > MAX_ATTRIBUTE_STRING:
        return redacted[:MAX_ATTRIBUTE_STRING] + "…"
    return redacted


def sanitize_attributes(value: Any, *, key: str | None = None, truncate: bool = True) -> Any:
    """Return JSON-safe, redacted values; full response diagnostics can opt out of size caps."""
    if key is not None and _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _redact_string(value, truncate=truncate)
    if isinstance(value, dict):
        items = list(value.items())
        if truncate:
            items = items[:MAX_ATTRIBUTE_ITEMS]
        return {
            str(item_key): sanitize_attributes(item_value, key=str(item_key), truncate=truncate)
            for item_key, item_value in items
        }
    if isinstance(value, (list, tuple, set, frozenset)):
        items = list(value)
        if truncate:
            items = items[:MAX_ATTRIBUTE_ITEMS]
        return [sanitize_attributes(item, truncate=truncate) for item in items]
    return _redact_string(str(value), truncate=truncate)


class TelemetryRecorder:
    """Append ordered, redacted events to a run's telemetry JSONL file."""

    def __init__(self, work_dir: Path | str, run_id: str, *, job_id: str | None = None) -> None:
        self.work_dir = Path(work_dir)
        self.path = self.work_dir / TELEMETRY_FILE
        self.run_id = run_id
        self.job_id = job_id
        self._state = _state_for(self.path)

    def emit(
        self,
        phase: str,
        event: str,
        *,
        status: str = "info",
        operation_id: str | None = None,
        parent_operation_id: str | None = None,
        duration_ms: float | None = None,
        attributes: dict[str, Any] | None = None,
        truncate_attributes: bool = True,
    ) -> dict[str, Any]:
        """Persist one event and return its serialized representation."""
        with self._state.lock:
            self._state.sequence += 1
            payload = {
                "schema_version": TELEMETRY_SCHEMA_VERSION,
                "sequence": self._state.sequence,
                "event_id": uuid.uuid4().hex,
                "timestamp": _utcnow_iso(),
                "run_id": self.run_id,
                "job_id": self.job_id,
                "phase": phase,
                "event": event,
                "status": status,
                "operation_id": operation_id,
                "parent_operation_id": parent_operation_id,
                "duration_ms": (round(duration_ms, 3) if duration_ms is not None else None),
                "attributes": sanitize_attributes(attributes or {}, truncate=truncate_attributes),
            }
            line = (json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
                "utf-8"
            )
            descriptor: int | None = None
            try:
                self.work_dir.mkdir(parents=True, exist_ok=True)
                descriptor = os.open(self.path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
                remaining = memoryview(line)
                while remaining:
                    written = os.write(descriptor, remaining)
                    if written == 0:
                        raise OSError("telemetry append wrote zero bytes")
                    remaining = remaining[written:]
            except OSError as exc:
                # A telemetry failure must not turn an otherwise recoverable run
                # into a failed run (for example, when trace volume fills a disk).
                logger.warning("could not append run telemetry to %s: %s", self.path, exc)
            finally:
                if descriptor is not None:
                    os.close(descriptor)
            return payload

    @contextmanager
    def span(
        self,
        phase: str,
        name: str,
        *,
        attributes: dict[str, Any] | None = None,
        parent_operation_id: str | None = None,
    ) -> Iterator[str]:
        """Record matched start/completion/failure events around an operation."""
        operation_id = uuid.uuid4().hex
        started = time.perf_counter()
        self.emit(
            phase,
            f"{name}.started",
            status="started",
            operation_id=operation_id,
            parent_operation_id=parent_operation_id,
            attributes=attributes,
        )
        try:
            yield operation_id
        except BaseException as exc:
            self.emit(
                phase,
                f"{name}.failed",
                status="failed",
                operation_id=operation_id,
                parent_operation_id=parent_operation_id,
                duration_ms=(time.perf_counter() - started) * 1_000,
                attributes={
                    **(attributes or {}),
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        else:
            self.emit(
                phase,
                f"{name}.completed",
                status="succeeded",
                operation_id=operation_id,
                parent_operation_id=parent_operation_id,
                duration_ms=(time.perf_counter() - started) * 1_000,
                attributes=attributes,
            )


def current_telemetry() -> TelemetryRecorder | None:
    """Return the recorder bound to the current execution context, if any."""
    return _current_recorder.get()


@contextmanager
def bind_telemetry(recorder: TelemetryRecorder) -> Iterator[TelemetryRecorder]:
    """Make ``recorder`` available to nested orchestration and callbacks."""
    token = _current_recorder.set(recorder)
    try:
        yield recorder
    finally:
        _current_recorder.reset(token)


def read_telemetry(
    work_dir: Path | str, *, after_sequence: int = 0, limit: int = 200
) -> list[dict[str, Any]]:
    """Read valid events after ``after_sequence`` in append order."""
    path = Path(work_dir) / TELEMETRY_FILE
    if not path.exists():
        return []
    events: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if int(event.get("sequence", 0)) <= after_sequence:
                    continue
                events.append(event)
                if len(events) >= limit:
                    break
    except OSError:
        return []
    return events
