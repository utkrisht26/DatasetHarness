"""Per-invocation search artifacts and progress across parallel tool batches."""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

RESEARCH_TOOLS = frozenset({"submit_search_plan", "web_search", "dataset_search", "sample_dataset"})


@dataclass
class _Batch:
    operation_id: str
    pending: set[str]
    research_attempted: bool
    productive: bool = False


@dataclass(frozen=True)
class BatchProgress:
    operation_id: str
    consecutive_unproductive: int
    action: str  # continue, review, or stop


@dataclass
class DiscoveryResearchState:
    """Shared by parent/subagents; network work never runs under the state lock."""

    _lock: Any = field(default_factory=Lock, repr=False)
    _artifacts: dict[str, int] = field(default_factory=dict)
    _reviewed: set[str] = field(default_factory=set)
    _empty_queries: dict[tuple[str, str], str] = field(default_factory=dict)
    _completed_search_tools: set[str] = field(default_factory=set)
    _query_locks: dict[tuple[str, str], Any] = field(default_factory=dict, repr=False)
    _batches: dict[str, _Batch] = field(default_factory=dict, repr=False)
    _consecutive: int = 0
    _review_grace_used: bool = False

    def query_lock(self, tool: str, query: str) -> Any:
        with self._lock:
            return self._query_locks.setdefault((tool, query.strip()), Lock())

    def empty_search(self, tool: str, query: str) -> str | None:
        with self._lock:
            return self._empty_queries.get((tool, query.strip()))

    def record_search(self, tool: str, query: str, path: str, count: int) -> None:
        with self._lock:
            self._artifacts[path] = count
            self._completed_search_tools.add(tool)
            if count == 0:
                self._empty_queries[(tool, query.strip())] = path

    def missing_searches(self, required_tools: set[str]) -> list[str]:
        """Blocked/failed calls and submitting a plan are not completed searches."""
        with self._lock:
            return sorted(required_tools - self._completed_search_tools)

    def review_hint(self) -> str:
        with self._lock:
            pending = sorted(
                (
                    (path, count)
                    for path, count in self._artifacts.items()
                    if count > 0 and path not in self._reviewed
                ),
                key=lambda item: (-item[1], item[0]),
            )
        if not pending:
            return ""
        files = "; ".join(f"/{path} ({count} result(s))" for path, count in pending[:8])
        return (
            f"Unreviewed search results: {files}. Read these nonempty files with read_file "
            "before searching again. Skip zero-result files; file numbers are not relevance ranks. "
            "Use the available candidates to finish your report, noting any coverage gaps."
        )

    def start_batch(self, operation_id: str, calls: list[dict[str, Any]]) -> None:
        tracked = [call for call in calls if call["name"] in RESEARCH_TOOLS | {"read_file"}]
        if not tracked:
            return
        batch = _Batch(
            operation_id,
            {call["id"] for call in tracked},
            any(call["name"] in RESEARCH_TOOLS for call in tracked),
        )
        with self._lock:
            for call_id in batch.pending:
                self._batches[call_id] = batch

    def complete_tool(
        self,
        call: dict[str, Any],
        *,
        unproductive: bool,
        failed: bool,
        limit: int,
    ) -> BatchProgress | None:
        """Evaluate only completed batches, independent of sibling finish order."""
        with self._lock:
            batch = self._batches.pop(call["id"], None)
            if batch is None:
                return None
            if call["name"] in RESEARCH_TOOLS and not unproductive:
                batch.productive = True
            if call["name"] == "read_file" and not failed:
                path = posixpath.normpath(str(call.get("args", {}).get("file_path", ""))).lstrip(
                    "/"
                )
                if self._artifacts.get(path, 0) > 0 and path not in self._reviewed:
                    self._reviewed.add(path)
                    batch.productive = True
            batch.pending.discard(call["id"])
            if batch.pending:
                return None
            if batch.productive:
                self._consecutive = 0
            elif batch.research_attempted:
                self._consecutive += 1
            consecutive = self._consecutive
            action = "continue"
            if limit > 0 and self._consecutive >= limit:
                unread = any(
                    count > 0 and path not in self._reviewed
                    for path, count in self._artifacts.items()
                )
                if unread and not self._review_grace_used:
                    # One bounded recovery window; ignored evidence cannot keep
                    # the paid loop alive indefinitely.
                    self._review_grace_used = True
                    self._consecutive = 0
                    action = "review"
                else:
                    action = "stop"
            return BatchProgress(batch.operation_id, consecutive, action)
