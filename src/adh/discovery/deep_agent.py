"""deepagents-backed Source Discovery — the research loop as the agent (T4).

:class:`DeepAgentDiscoveryAgent` drives discovery with ``create_deep_agent``
(locked decision 3): the existing :class:`SearchProvider` / :class:`DatasetRegistry`
ports are wrapped as plain-function tools (no MCP server), a ``web-scout``
subagent burns its own context on raw search triage, and the loop terminates by
emitting a structured :class:`DiscoveryReport` (``ToolStrategy``) which is
converted into the same :class:`FetchPlan` the heuristic agent produces.

The split of judgement is deliberate: the model judges shape / quality / yield;
license, freshness, domain policy, scoring, and ranking stay harness-side and
deterministic (:mod:`adh.discovery.agent` shared verdicts).

Bounds & metering (decision 4): the LangGraph ``recursion_limit`` caps runaway
tool loops, and every model call is recorded into a :class:`UsageTracker` via
LangChain's ``UsageMetadataCallbackHandler`` — even when the run errors, tokens
already spent are folded in.

:class:`FallbackDiscoveryAgent` wraps any primary with the heuristic
:class:`SourceDiscoveryAgent`: on agent error (missing key, transport failure,
junk output, recursion cap) discovery still returns a plan, degraded and noted.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from threading import Lock
from typing import Annotated, Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from adh.contract import DiscoveryMode, JobSpec, SourceChannel
from adh.llm import TokenUsage, UsageTracker
from adh.llm.response_logging import (
    bind_model_response_operation,
    capture_model_responses,
    install_response_logging,
    log_model_response,
)
from adh.llm.prompts import (
    DISCOVERY_AGENT_SYSTEM,
    DISCOVERY_AGENT_TASK,
    DISCOVERY_WEB_SCOUT,
)
from adh.search import (
    DatasetRegistry,
    InMemoryDatasetRegistry,
    InMemorySearchProvider,
    SearchProvider,
)
from adh.telemetry import TelemetryRecorder, current_telemetry

from .agent import (
    DiscoveryAgent,
    DiscoveryConfig,
    SourceDiscoveryAgent,
    dedupe_and_rank,
    domain_allowed,
    fixed_plan,
    freshness_ok,
    goal_text,
    license_ok,
    score_candidate,
    target_shape,
)
from .models import Aspect, CandidateSource, FetchPlan, RankedSource, SourceEvaluation
from .research import RESEARCH_TOOLS, DiscoveryResearchState

logger = logging.getLogger(__name__)

# Generous headroom for decompose -> per-aspect search -> sample -> judge, but a
# hard stop for a model stuck re-calling tools (plan §failure-modes).
DEFAULT_RECURSION_LIMIT = 1500

# An upstream provider can return a well-formed-but-EMPTY completion (HTTP 200,
# no text, no tool calls, not even usage metadata — observed live via OpenRouter
# routing). That is a transient upstream fault, not a verdict on the task, so the
# whole research invocation is retried this many times before giving up.
EMPTY_COMPLETION_RETRIES = 2

# Keep the existing research context while correcting malformed final reports.
# ToolStrategy otherwise retries schema failures until the graph/budget cap.
REPORT_VALIDATION_RETRIES = 2

# Keep the legacy option name, but count completed research batches: concurrent
# searches with any hits are productive regardless of completion order. Unread
# hits get one bounded review window before FallbackDiscoveryAgent takes over.
DEFAULT_MAX_CONSECUTIVE_UNPRODUCTIVE_TOOLS = 3

_UNPRODUCTIVE_TOOL_OUTPUT = re.compile(
    r"^(?:no sample content available|invalid dataset id|search blocked:|"
    r"search plan rejected:|search skipped:|(?:Dataset search|Search) results for query .* "
    r"\(found 0 results\) have been saved)",
    re.IGNORECASE | re.DOTALL,
)
_RESEARCH_TOOLS = RESEARCH_TOOLS
_INVALID_DATASET_ID = re.compile(r"[\s{}\[\]'\"]")
_USEFUL_QUERY = re.compile(r"[^\W_]")


class DiscoveryAgentError(RuntimeError):
    """The deep agent could not produce a structured discovery report."""


class DiscoveryBudgetExceeded(RuntimeError):
    """The discovery token budget was exhausted before a report was produced.

    Raised between model turns (never mid-call), so every token already spent is
    metered and folded into the run's :class:`UsageTracker`.
    """


class DiscoveryNoProgress(RuntimeError):
    """The agent exhausted its allowance of unproductive research batches."""


@dataclass
class DiscoveryPlanningGate:
    """Shared state that prevents research tools running before an accepted plan."""

    allowed_channels: frozenset[SourceChannel]
    submitted: bool = False
    plan: dict[str, Any] | None = None


# ------------------------------------------------------- the agent's wire format
class ReportAspect(BaseModel):
    """One sub-topic of the goal, as reported by the agent."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    topic: str = Field(
        min_length=1,
        max_length=500,
        description="Short topic label, not serialized JSON or the report body",
    )
    rationale: str = ""
    keywords: list[str] = Field(default_factory=list)


class ReportSource(BaseModel):
    """One judged candidate source, tied to its aspect by index.

    A ``web`` source is a crawl SEED: acquisition starts at ``locator`` and expands
    along on-topic links, steering by ``title`` + ``snippet``. The field
    descriptions below ship to the model as the structured-output schema.
    """

    channel: str = Field(
        "web", description="'web' (page to crawl) or 'datasets' (existing dataset)"
    )
    locator: str = Field(
        description=(
            "For web: the seed URL the crawler starts from (prefer hub/index pages). "
            "For datasets: the dataset id."
        )
    )
    title: str = Field("", description="Descriptive title; steers the crawler's link-following")
    aspect_index: int = Field(0, description="Index into the report's aspects list")
    snippet: str = Field(
        "",
        description="Keyword-rich summary of what lives there; steers the crawl frontier",
    )
    license: str | None = Field(None, description="License if visible (e.g. 'MIT'), else null")
    published: str | None = Field(None, description="Publish/update date if visible (YYYY-MM-DD)")
    discovered_via: str = Field(
        "",
        description="The search query that surfaced it, or 'model-knowledge' for proposed seeds",
    )
    shape_match: float = Field(0.0, description="0-1: does it contain records of the target shape?")
    quality: float = Field(0.0, description="0-1: signal vs noise of the content")
    est_yield: int = Field(0, description="Usable records this source could plausibly produce")
    notes: str = Field("", description="One-line judgement rationale")


class DiscoveryReport(BaseModel):
    """The deep agent's final structured output — everything a FetchPlan needs."""

    model_config = ConfigDict(extra="forbid")

    aspects: list[ReportAspect] = Field(min_length=1)
    sources: list[ReportSource] = Field(
        description=(
            "Candidate source objects at the top level, alongside aspects. "
            "Always supply this array; use [] only when there are no suitable external sources."
        )
    )
    synthesis_aspect_indexes: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


def _empty_report_feedback(
    report: DiscoveryReport,
    planning_gate: DiscoveryPlanningGate,
    research: DiscoveryResearchState,
) -> str | None:
    """An empty source list needs research evidence or complete synthesis routing."""
    if report.sources:
        return None
    if not planning_gate.submitted:
        return "Submit a concrete search plan with submit_search_plan, then execute it."
    allowed = planning_gate.allowed_channels
    if SourceChannel.SYNTHESIS in allowed and set(report.synthesis_aspect_indexes) == set(
        range(len(report.aspects))
    ):
        return None

    channel_tools = {
        "web_search": SourceChannel.WEB,
        "dataset_search": SourceChannel.DATASETS,
    }
    required_tools = {tool for tool, channel in channel_tools.items() if channel in allowed}
    if not required_tools:
        return (
            "No external search channel is enabled. An empty source list requires every "
            "aspect to be routed to synthesis, and only when synthesis is allowed."
        )
    missing = research.missing_searches(required_tools)
    if missing:
        queries = (planning_gate.plan or {}).get("queries", {})
        calls = []
        for tool in missing:
            planned_queries = queries.get(channel_tools[tool].value, [])
            calls.append(
                f"{tool} with {json.dumps({'query': planned_queries[0]}, ensure_ascii=False)}"
                if planned_queries
                else f"{tool} with a concrete query about the dataset goal"
            )
        return (
            "Cannot conclude that no external sources exist before executing searches. "
            "A submitted plan or a blocked/failed call does not count as research. Execute "
            + "; ".join(calls)
            + ". Review nonempty results with read_file and report suitable candidates. "
            "If none qualify, explain the coverage gap in notes."
        )
    if hint := research.review_hint():
        return (
            "Cannot return sources=[] while search results remain unreviewed. "
            + hint
            + " Report suitable candidates, or explain in notes why the reviewed hits do not qualify."
        )
    if not any(note.strip() for note in report.notes):
        return (
            "An empty source list must explain in notes why completed searches and reviewed "
            "candidates yielded no suitable external sources. Include candidates if any qualify."
        )
    return None


# ------------------------------------------------------------- tools over ports
def build_discovery_tools(
    search: SearchProvider,
    registry: DatasetRegistry,
    work_dir: Path,
    *,
    limit: int = 8,
    budget_status: Callable[[], str] | None = None,
    allowed_channels: set[SourceChannel] | frozenset[SourceChannel] | None = None,
    planning_gate: DiscoveryPlanningGate | None = None,
    research_state: DiscoveryResearchState | None = None,
) -> list:
    """Wrap the existing search/registry ports as agent tools (no MCP).

    ``budget_status`` optionally supplies a one-line spend summary appended to
    every tool result — the soft nudge that lets the model wrap up on its own
    before the hard :class:`DiscoveryBudgetExceeded` guard fires.
    """

    counters = {"web": 0, "dataset": 0}
    counter_lock = Lock()
    plan_execution_lock = Lock()
    executed_plan_queries: dict[str, set[str]] = {"web": set(), "datasets": set()}
    research = research_state if research_state is not None else DiscoveryResearchState()
    enabled = set(allowed_channels or {SourceChannel.WEB, SourceChannel.DATASETS})

    def _nudge(text: str) -> str:
        if hint := research.review_hint():
            text += "\n\n" + hint
        if budget_status is None:
            return text
        line = budget_status()
        return f"{text}\n\n[budget] {line}" if line else text

    def _clean(values: list[str]) -> list[str]:
        return list(dict.fromkeys(value.strip() for value in values if value.strip()))

    def submit_search_plan(
        aspects: Annotated[
            list[str], Field(min_length=1, description="At least one concrete topic")
        ],
        web_queries: list[str] | None = None,
        dataset_queries: list[str] | None = None,
        rationale: str = "",
    ) -> str:
        """Submit the research plan before searching: aspects, queries, channels, and rationale."""
        clean_aspects = _clean(aspects)
        clean_web = _clean(web_queries or [])
        clean_datasets = _clean(dataset_queries or [])
        if planning_gate is None:
            return "Search planning is not enabled for this run."
        if not clean_aspects:
            return "Search plan rejected: include at least one concrete aspect."
        disabled: list[str] = []
        if clean_web and SourceChannel.WEB not in enabled:
            disabled.append("web")
        if clean_datasets and SourceChannel.DATASETS not in enabled:
            disabled.append("datasets")
        if disabled:
            return (
                "Search plan rejected: disabled channel(s) requested: "
                + ", ".join(disabled)
                + ". Use only the allowed channels from the task."
            )
        queries = clean_web + clean_datasets
        searchable = SourceChannel.WEB in enabled or SourceChannel.DATASETS in enabled
        if searchable and (
            not queries or any(_USEFUL_QUERY.search(query) is None for query in queries)
        ):
            return "Search plan rejected: include concrete, non-empty queries for enabled channels."

        payload = {
            "agent": "deep-agent",
            "aspects": clean_aspects,
            "queries": {"web": clean_web, "datasets": clean_datasets},
            "rationale": rationale.strip(),
            "allowed_channels": sorted(channel.value for channel in planning_gate.allowed_channels),
        }
        research_dir = Path(work_dir) / "research"
        research_dir.mkdir(parents=True, exist_ok=True)
        with (research_dir / "search_plan.json").open("w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2)
        with plan_execution_lock:
            for completed in executed_plan_queries.values():
                completed.clear()
        planning_gate.plan = payload
        planning_gate.submitted = True
        return _nudge(
            "Search plan accepted and saved to 'research/search_plan.json': "
            f"{len(clean_aspects)} aspect(s), {len(clean_web)} web query/queries, "
            f"{len(clean_datasets)} dataset query/queries. Execute this plan now."
        )

    def _claim_planned_query(channel: str) -> str | None:
        """Atomically reserve the next accepted query for a malformed tool call."""
        if planning_gate is None or not planning_gate.submitted or planning_gate.plan is None:
            return None
        with plan_execution_lock:
            completed = executed_plan_queries[channel]
            for candidate in planning_gate.plan["queries"][channel]:
                query = candidate.strip()
                if _USEFUL_QUERY.search(query) is not None and query not in completed:
                    completed.add(query)
                    return query
        return None

    def _mark_plan_query_executed(channel: str, query: str) -> None:
        with plan_execution_lock:
            executed_plan_queries[channel].add(query)

    def _search_guard(query: str, channel: str) -> str | None:
        if planning_gate is not None and not planning_gate.submitted:
            return _nudge(
                "Search blocked: call submit_search_plan first with concrete aspects, "
                "channel-specific queries, and a rationale."
            )
        if _USEFUL_QUERY.search(query) is None:
            message = (
                "Search blocked: replace the empty or placeholder query with a concrete query."
            )
            if planning_gate is not None and planning_gate.plan is not None:
                queries = planning_gate.plan["queries"][channel]
                if queries:
                    message += (
                        " Every query in the accepted plan for this channel has already been "
                        "executed; use a concrete new query or submit a revised plan."
                    )
            return _nudge(message)
        return None

    def _run_search(query: str, channel: str, tool_name: str, fetch: Callable) -> str:
        requested_query = query
        query = query.strip()
        repair_note = ""
        if _USEFUL_QUERY.search(query) is None:
            if replacement := _claim_planned_query(channel):
                query = replacement
                repair_note = (
                    f"Recovered invalid {tool_name} query {requested_query!r}; "
                    f"executed accepted-plan query {query!r} instead."
                )
        if blocked := _search_guard(query, channel):
            return blocked
        if not repair_note:
            _mark_plan_query_executed(channel, query)
        # Serialize only identical queries, leaving distinct searches parallel.
        with research.query_lock(tool_name, query):
            if previous := research.empty_search(tool_name, query):
                return _nudge(
                    f"Search skipped: unchanged query {query!r} already returned 0 results "
                    f"in '{previous}'. Do not repeat it. Review nonempty results, broaden the "
                    "query or remove the site: restriction, or finish with a coverage gap."
                )
            results = fetch(query)
            counter = "web" if channel == "web" else "dataset"
            with counter_lock:
                counters[counter] += 1
                rel_path = f"research/{tool_name}_{counters[counter]}.json"
            path = Path(work_dir) / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("w", encoding="utf-8") as stream:
                json.dump({"query": query, "results": results}, stream, indent=2)
            research.record_search(tool_name, query, rel_path, len(results))
        prefix = "Search" if channel == "web" else "Dataset search"
        guidance = (
            f"Read '{rel_path}' with read_file to identify candidates."
            if results
            else "This file is empty; skip reading it. Review other nonempty results or change the query."
        )
        result = (
            f"{prefix} results for query '{query}' (found {len(results)} results) "
            f"have been saved to virtual path '{rel_path}'. {guidance}"
        )
        if repair_note:
            # Keep the canonical search-result prefix first: the no-progress
            # classifier recognizes zero-result searches from that prefix.
            result = f"{result}\n\n{repair_note}"
        return _nudge(result)

    def _web_results(query: str) -> list[dict[str, Any]]:
        return [
            {
                "url": hit.url,
                "title": hit.title,
                "snippet": hit.snippet,
                "license_hint": hit.license_hint,
                "published": hit.published,
            }
            for hit in search.search(query, limit=limit)
        ]

    def _dataset_results(query: str) -> list[dict[str, Any]]:
        return [
            {
                "dataset_id": hit.dataset_id,
                "locator": hit.locator,
                "title": hit.title,
                "description": hit.description,
                "license": hit.license,
                "downloads": hit.downloads,
                "last_modified": hit.last_modified,
            }
            for hit in registry.search(query, limit=limit)
        ]

    def web_search(query: str) -> str:
        """Search candidate pages with a concrete query; do not repeat queries that returned no hits."""
        return _run_search(query, "web", "web_search", _web_results)

    def dataset_search(query: str) -> str:
        """Search dataset registries with a concrete query; do not repeat queries that returned no hits."""
        return _run_search(query, "datasets", "dataset_search", _dataset_results)

    def sample_dataset(dataset_id: str) -> str:
        """Fetch a small content sample from a dataset to judge shape/quality (empty if unavailable)."""
        if planning_gate is not None and not planning_gate.submitted:
            return (
                "Search blocked: call submit_search_plan first with concrete aspects, "
                "channel-specific queries, and a rationale."
            )
        if not dataset_id or _INVALID_DATASET_ID.search(dataset_id):
            return (
                f"Invalid dataset id '{dataset_id}'. Use the exact `dataset_id` value "
                "from a dataset_search result; do not add JSON punctuation or whitespace."
            )
        sample = registry.sample(dataset_id) or ""
        if not sample:
            return f"No sample content available for dataset '{dataset_id}'."

        safe_id = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in dataset_id)
        filename = f"dataset_sample_{safe_id}.txt"
        rel_path = f"research/{filename}"
        abs_dir = Path(work_dir) / "research"
        abs_dir.mkdir(parents=True, exist_ok=True)
        abs_path = abs_dir / filename

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(sample)

        return _nudge(
            f"Sample content for dataset '{dataset_id}' has been saved to virtual path "
            f"'{rel_path}'. "
            f"You MUST use the `read_file` tool to view the content of '{rel_path}'."
        )

    tools = [submit_search_plan] if planning_gate is not None else []
    if SourceChannel.WEB in enabled:
        tools.append(web_search)
    if SourceChannel.DATASETS in enabled:
        tools.extend([dataset_search, sample_dataset])
    return tools


def _short(text: object, limit: int = 160) -> str:
    """One-line, length-capped view of a value for a log line."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _assistant_said_nothing(message: Any) -> bool:
    """True when a reply carries neither text nor tool calls (an empty completion).

    Duck-typed on the message shape so this module stays importable without
    LangChain installed; anything that is not an AIMessage is not "nothing".
    """
    if message is None or getattr(message, "tool_calls", None):
        return False
    if type(message).__name__ != "AIMessage":
        return False
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return not content.strip()
    # content can also be a list of typed blocks; any non-empty text counts as speech
    return not any(
        (isinstance(block, str) and block.strip())
        or (isinstance(block, dict) and str(block.get("text") or "").strip())
        for block in content
    )


def _describe_final_turn(message: Any) -> str:
    """One-line diagnosis of why the final turn produced no report."""
    if _assistant_said_nothing(message):
        return (
            "final model turn was an empty completion "
            "(no text, no tool calls — likely a transient upstream/provider issue)"
        )
    if message is None:
        return "no messages returned"
    reply = getattr(message, "content", "")
    return f"final model turn replied in prose ({_short(reply)}) without emitting a DiscoveryReport"


def _build_tracer(
    tracker: UsageTracker | None,
    budget: int | None = None,
    *,
    telemetry: TelemetryRecorder | None = None,
    parent_operation_id: str | None = None,
    max_consecutive_unproductive_tools: int = DEFAULT_MAX_CONSECUTIVE_UNPRODUCTIVE_TOOLS,
    research_state: DiscoveryResearchState | None = None,
):
    """Meter tokens, trace steps, and record stops for the agent middleware to enforce.

    Three jobs in one handler so there is a single ``on_llm_end`` pass:

    * **Metering** (always, when a ``tracker`` is given): folds each reply's
      usage — including provider-surfaced cost — into the run's
      :class:`UsageTracker`, so spend is captured even when the loop later errors
      or hits the recursion cap (decision 4: cost is real or unknown).
    * **Guards**: record budget exhaustion and repeated unproductive research.
      Middleware raises after the callback completes, so an expected stop is
      not reported as a callback malfunction. Every spent token stays metered.
    * **Tracing** (always): logs each tool call, model turn with a running token
      total, and subagent step to the ``adh.discovery`` logger. These are INFO/
      DEBUG, so they surface only when logging is configured (``adh --verbose`` /
      ``adh serve``) and cost nothing otherwise. This is what turns "40k tokens
      then an error with no trace" into a step-by-step record of what ran.

    The instance carries live ``turns`` / ``tool_calls`` counters the caller reads
    for the start/finish/error summary lines.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    research = research_state if research_state is not None else DiscoveryResearchState()

    class _AgentTracer(BaseCallbackHandler):
        # Unexpected telemetry/metering errors must still propagate.
        raise_error = True

        def __init__(self) -> None:
            self.turns = 0
            self.tool_calls = 0
            self.consecutive_unproductive_tools = 0
            self._stop: RuntimeError | None = None
            self._tool_started: dict[str, float] = {}
            self._tool_names: dict[str, str] = {}
            self._model_started: dict[str, float] = {}

        def raise_if_stopped(self) -> None:
            if self._stop is not None:
                raise self._stop

        def record_tool_result(self, call, output) -> None:
            # Middleware sees validation errors too, including those handled by
            # ToolNode before on_tool_end can run.
            update = getattr(output, "update", None)
            if isinstance(update, dict) and update.get("messages"):
                output = update["messages"][-1]
            output_text = str(getattr(output, "content", output))
            failed = getattr(output, "status", None) == "error" or output_text.startswith("Error:")
            unproductive = failed or _UNPRODUCTIVE_TOOL_OUTPUT.search(output_text) is not None
            progress = research.complete_tool(
                call,
                unproductive=unproductive,
                failed=failed,
                limit=max_consecutive_unproductive_tools,
            )
            if progress is None:
                return
            self.consecutive_unproductive_tools = progress.consecutive_unproductive
            attributes = {
                "consecutive_unproductive": progress.consecutive_unproductive,
                "limit": max_consecutive_unproductive_tools,
                "tool": call["name"],
                "output": output_text,
            }
            if telemetry is not None:
                telemetry.emit(
                    "discovery",
                    "research.batch.completed",
                    operation_id=progress.operation_id,
                    attributes={**attributes, "action": progress.action},
                )
            if progress.action == "review":
                hint = research.review_hint()
                logger.info(
                    "deep-agent recovery window for reviewing existing results: %s",
                    hint,
                )
                if telemetry is not None:
                    telemetry.emit(
                        "discovery",
                        "guard.review_pending",
                        operation_id=progress.operation_id,
                        attributes={**attributes, "guidance": hint},
                    )
            elif progress.action == "stop":
                if self._stop is None:
                    self._stop = DiscoveryNoProgress(
                        f"discovery stopped after {progress.consecutive_unproductive} consecutive "
                        f"unproductive research batches (last tool: {call['name']}; "
                        f"{_short(output_text)})"
                    )
                if telemetry is not None:
                    telemetry.emit(
                        "discovery",
                        "guard.no_progress",
                        status="failed",
                        operation_id=progress.operation_id,
                        attributes=attributes,
                    )

        @staticmethod
        def _operation_id(kwargs) -> str:
            value = kwargs.get("run_id")
            return str(value) if value is not None else uuid.uuid4().hex

        @staticmethod
        def _parent_id(kwargs) -> str | None:
            value = kwargs.get("parent_run_id")
            return str(value) if value is not None else parent_operation_id

        def _model_start(self, serialized, input_count: int, kwargs) -> None:
            operation_id = self._operation_id(kwargs)
            bind_model_response_operation(operation_id, self._parent_id(kwargs), telemetry)
            self._model_started[operation_id] = time.perf_counter()
            if telemetry is not None:
                model_id = (serialized or {}).get("id") or (serialized or {}).get("name")
                telemetry.emit(
                    "discovery",
                    "model.started",
                    status="started",
                    operation_id=operation_id,
                    parent_operation_id=self._parent_id(kwargs),
                    attributes={
                        "model": model_id,
                        "model_name": (kwargs.get("invocation_params") or {}).get("model_name")
                        or (kwargs.get("invocation_params") or {}).get("model"),
                        "input_batches": input_count,
                    },
                )

        def on_chat_model_start(self, serialized, messages, **kwargs) -> None:
            self._model_start(serialized, len(messages), kwargs)

        def on_llm_start(self, serialized, prompts, **kwargs) -> None:
            self._model_start(serialized, len(prompts), kwargs)

        def on_tool_start(self, serialized, input_str, **kwargs) -> None:
            self.tool_calls += 1
            name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
            operation_id = self._operation_id(kwargs)
            self._tool_started[operation_id] = time.perf_counter()
            self._tool_names[operation_id] = name
            phase = "planning" if name == "submit_search_plan" else "discovery"
            logger.info("deep-agent → tool %s(%s)", name, _short(input_str))
            if telemetry is not None:
                telemetry.emit(
                    phase,
                    "tool.started",
                    status="started",
                    operation_id=operation_id,
                    parent_operation_id=self._parent_id(kwargs),
                    attributes={"tool": name, "input": input_str},
                )

        def on_tool_end(self, output, **kwargs) -> None:
            logger.debug("deep-agent ← %s", _short(output))
            operation_id = self._operation_id(kwargs)
            started = self._tool_started.pop(operation_id, None)
            name = self._tool_names.pop(operation_id, "tool")
            phase = "planning" if name == "submit_search_plan" else "discovery"
            output_text = str(getattr(output, "content", output))
            failed = getattr(output, "status", None) == "error"
            unproductive = name in _RESEARCH_TOOLS and (
                failed or _UNPRODUCTIVE_TOOL_OUTPUT.search(output_text) is not None
            )
            if telemetry is not None:
                telemetry.emit(
                    phase,
                    "tool.completed",
                    status=("failed" if failed else "empty" if unproductive else "succeeded"),
                    operation_id=operation_id,
                    parent_operation_id=self._parent_id(kwargs),
                    duration_ms=((time.perf_counter() - started) * 1_000 if started else None),
                    attributes={
                        "tool": name,
                        "output": output_text,
                    },
                )

        def on_tool_error(self, error, **kwargs) -> None:
            logger.warning("deep-agent ✗ tool error: %s", _short(error))
            operation_id = self._operation_id(kwargs)
            started = self._tool_started.pop(operation_id, None)
            name = self._tool_names.pop(operation_id, "tool")
            phase = "planning" if name == "submit_search_plan" else "discovery"
            if telemetry is not None:
                telemetry.emit(
                    phase,
                    "tool.failed",
                    status="failed",
                    operation_id=operation_id,
                    parent_operation_id=self._parent_id(kwargs),
                    duration_ms=((time.perf_counter() - started) * 1_000 if started else None),
                    attributes={
                        "tool": name,
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )

        def on_llm_end(self, response, **kwargs) -> None:
            log_model_response(
                "parsed",
                response,
                telemetry=telemetry,
                operation_id=self._operation_id(kwargs),
                parent_operation_id=self._parent_id(kwargs),
            )
            delta = 0
            input_tokens = 0
            output_tokens = 0
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    if message is None:
                        continue
                    research.start_batch(
                        self._operation_id(kwargs), getattr(message, "tool_calls", [])
                    )
                    self.turns += 1
                    usage = (
                        tracker.add_message(message)
                        if tracker is not None
                        else TokenUsage.from_message(message)
                    )
                    delta += usage.total_tokens
                    input_tokens += usage.input_tokens
                    output_tokens += usage.output_tokens
            total = tracker.total.total_tokens if tracker is not None else delta
            logger.info(
                "deep-agent · model turn %d (+%d tok, %d total)",
                self.turns,
                delta,
                total,
            )
            operation_id = self._operation_id(kwargs)
            started = self._model_started.pop(operation_id, None)
            if telemetry is not None:
                telemetry.emit(
                    "discovery",
                    "model.completed",
                    status="succeeded",
                    operation_id=operation_id,
                    parent_operation_id=self._parent_id(kwargs),
                    duration_ms=((time.perf_counter() - started) * 1_000 if started else None),
                    attributes={
                        "turn": self.turns,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "total_tokens": delta,
                        "run_total_tokens": total,
                    },
                )
            if budget is not None and tracker is not None and total >= budget:
                if telemetry is not None:
                    telemetry.emit(
                        "discovery",
                        "guard.token_budget",
                        status="failed",
                        operation_id=operation_id,
                        parent_operation_id=self._parent_id(kwargs),
                        attributes={"used_tokens": total, "budget_tokens": budget},
                    )
                self._stop = DiscoveryBudgetExceeded(
                    f"discovery token budget exhausted: {total}/{budget} tokens used"
                )

        def on_llm_error(self, error, **kwargs) -> None:
            if kwargs.get("response") is not None:
                log_model_response(
                    "parsed",
                    kwargs["response"],
                    telemetry=telemetry,
                    operation_id=self._operation_id(kwargs),
                    parent_operation_id=self._parent_id(kwargs),
                    status="failed",
                )
            logger.warning("deep-agent ✗ model error: %s", _short(error))
            operation_id = self._operation_id(kwargs)
            started = self._model_started.pop(operation_id, None)
            if telemetry is not None:
                telemetry.emit(
                    "discovery",
                    "model.failed",
                    status="failed",
                    operation_id=operation_id,
                    parent_operation_id=self._parent_id(kwargs),
                    duration_ms=((time.perf_counter() - started) * 1_000 if started else None),
                    attributes={
                        "error_type": type(error).__name__,
                        "error": str(error),
                    },
                )

    return _AgentTracer()


# ------------------------------------------------------------------- the agent
@dataclass
class DeepAgentDiscoveryAgent:
    """Discovery driven by a deepagents research loop over the existing ports.

    ``model`` is a LangChain chat model or a ``"<provider>:<model>"`` id (``None``
    uses :attr:`adh.settings.Settings.discovery_model`). Everything is injectable
    for offline tests: the in-memory ports feed the tools, and a scripted fake
    chat model drives the very same ``create_deep_agent`` loop with no network.
    """

    model: Any = None
    search: SearchProvider = field(default_factory=InMemorySearchProvider)
    registry: DatasetRegistry = field(default_factory=InMemoryDatasetRegistry)
    config: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    usage: UsageTracker | None = None
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    token_budget: int | None = None  # None -> derive from spec.limits.max_llm_tokens
    max_consecutive_unproductive_tools: int = DEFAULT_MAX_CONSECUTIVE_UNPRODUCTIVE_TOOLS

    def discover(self, spec: JobSpec, work_dir: Path) -> FetchPlan:
        job_id = spec.job_id or spec.name
        if spec.discovery.mode == DiscoveryMode.FIXED:
            # Nothing to research: the user gave exact sources.
            return fixed_plan(spec, job_id, [])

        report = self._run_agent(spec, job_id, work_dir)
        return self._to_plan(report, spec, job_id)

    def _effective_budget(self, spec: JobSpec) -> int | None:
        """Explicit ``token_budget`` wins; otherwise the spec's LLM ceiling."""
        if self.token_budget is not None:
            return max(int(self.token_budget), 0)
        limit = getattr(spec.limits, "max_llm_tokens", None)
        return int(limit) if limit else None

    # ---------------------------------------------------------------- research
    def _run_agent(self, spec: JobSpec, job_id: str, work_dir: Path) -> DiscoveryReport:
        telemetry = current_telemetry()
        if telemetry is None:
            return self._run_agent_impl(spec, job_id, work_dir)

        attributes: dict[str, Any] = {
            "recursion_limit": self.recursion_limit,
            "token_budget": self._effective_budget(spec),
            "channels": [channel.value for channel in spec.discovery.channels],
        }
        with telemetry.span("discovery", "agent", attributes=attributes) as operation_id:
            report = self._run_agent_impl(
                spec,
                job_id,
                work_dir,
                telemetry=telemetry,
                parent_operation_id=operation_id,
            )
            attributes.update(
                {
                    "aspects": len(report.aspects),
                    "sources": len(report.sources),
                    "llm": self.usage.to_dict() if self.usage is not None else None,
                }
            )
            return report

    def _run_agent_impl(
        self,
        spec: JobSpec,
        job_id: str,
        work_dir: Path,
        *,
        telemetry: TelemetryRecorder | None = None,
        parent_operation_id: str | None = None,
    ) -> DiscoveryReport:
        budget = self._effective_budget(spec)
        research_state = DiscoveryResearchState()
        task = DISCOVERY_AGENT_TASK.format(
            goal=goal_text(spec),
            target_shape=target_shape(spec),
            channels=", ".join(c.value for c in spec.discovery.channels),
            constraints=self._constraints_text(spec),
        )
        # One handler meters tokens/cost as each model call ends, traces every
        # tool call / model turn, AND enforces the token budget — so the run is
        # observable live and its spend is captured even when the loop errors,
        # exhausts its budget, or hits the recursion cap.
        tracer = _build_tracer(
            self.usage,
            budget=budget,
            telemetry=telemetry,
            parent_operation_id=parent_operation_id,
            max_consecutive_unproductive_tools=self.max_consecutive_unproductive_tools,
            research_state=research_state,
        )
        graph, planning_gate = self._build_graph(
            spec,
            work_dir,
            budget=budget,
            check_stop=tracer.raise_if_stopped,
            record_tool_result=tracer.record_tool_result,
            research_state=research_state,
            telemetry=telemetry,
            parent_operation_id=parent_operation_id,
        )
        logger.info(
            "deep-agent starting: job=%s goal=%r channels=%s recursion_limit=%d budget=%s",
            job_id,
            _short(goal_text(spec), 80),
            ",".join(c.value for c in spec.discovery.channels),
            self.recursion_limit,
            budget,
        )
        report: DiscoveryReport | None = None
        last_message: Any = None
        attempts = 1 + EMPTY_COMPLETION_RETRIES
        for attempt in range(1, attempts + 1):
            try:
                result = graph.invoke(
                    {"messages": [{"role": "user", "content": task}]},
                    config={
                        "recursion_limit": self.recursion_limit,
                        "callbacks": [tracer],
                    },
                )
            except (
                DiscoveryNoProgress,
                DiscoveryBudgetExceeded,
                DiscoveryAgentError,
            ) as exc:
                logger.warning(
                    "deep-agent stopped after %d model turn(s), %d tool call(s), %d token(s): %s",
                    tracer.turns,
                    tracer.tool_calls,
                    self.usage.total.total_tokens if self.usage is not None else 0,
                    exc,
                )
                raise
            except Exception:
                # Surface the full traceback and the spend at the point of failure,
                # then re-raise for FallbackDiscoveryAgent to degrade the run.
                logger.exception(
                    "deep-agent FAILED after %d model turn(s), %d tool call(s), %d token(s)",
                    tracer.turns,
                    tracer.tool_calls,
                    self.usage.total.total_tokens if self.usage is not None else 0,
                )
                raise

            report = result.get("structured_response") if isinstance(result, dict) else None
            if isinstance(report, DiscoveryReport):
                break
            messages = result.get("messages") if isinstance(result, dict) else []
            last_message = messages[-1] if messages else None
            if attempt < attempts and _assistant_said_nothing(last_message):
                # An empty completion is a transient upstream fault, not an answer:
                # rerun the research from scratch (the budget guard still bounds it).
                logger.warning(
                    "deep-agent turn produced an empty completion (no text, no tool "
                    "calls); retrying the research (%d/%d)",
                    attempt,
                    EMPTY_COMPLETION_RETRIES,
                )
                continue
            break

        spent = self.usage.total.total_tokens if self.usage is not None else 0
        if isinstance(report, DiscoveryReport) and not planning_gate.submitted:
            raise DiscoveryAgentError(
                "deep agent emitted a DiscoveryReport before submitting its search plan"
            )
        if not isinstance(report, DiscoveryReport):
            if _assistant_said_nothing(last_message):
                # Empties on every attempt is the signature of a model whose
                # upstream silently returns nothing for tool-bound requests
                # (observed live: OpenRouter/Stealth serves tool calls as empty
                # 200s) — say so, because the remedy is a different model.
                reason = (
                    "every research attempt returned an empty completion "
                    "(no text, no tool calls, no usage) — the configured discovery "
                    "model's provider appears not to support tool calling; "
                    "configure a different discovery model"
                )
            else:
                reason = _describe_final_turn(last_message)
            logger.error(
                "deep-agent produced no structured report after %d turn(s), %d token(s): %s",
                tracer.turns,
                spent,
                reason,
            )
            raise DiscoveryAgentError(
                f"deep agent produced no structured DiscoveryReport ({reason})"
            )
        if not report.aspects:
            raise DiscoveryAgentError("deep agent report contained no aspects")
        logger.info(
            "deep-agent finished: %d aspect(s), %d source(s) — %d turn(s), %d tool call(s), %d token(s)",
            len(report.aspects),
            len(report.sources),
            tracer.turns,
            tracer.tool_calls,
            spent,
        )
        return report

    def _build_graph(
        self,
        spec: JobSpec,
        work_dir: Path,
        *,
        budget: int | None = None,
        check_stop: Callable[[], None] | None = None,
        record_tool_result: Callable[[dict, Any], None] | None = None,
        research_state: DiscoveryResearchState | None = None,
        telemetry: TelemetryRecorder | None = None,
        parent_operation_id: str | None = None,
    ):
        try:
            from deepagents import create_deep_agent
            from deepagents.backends import FilesystemBackend
            from deepagents.middleware.subagents import GENERAL_PURPOSE_SUBAGENT
            from langchain.agents.middleware import AgentMiddleware
            from langchain.agents.structured_output import ToolStrategy
            from langchain_core.messages import SystemMessage
        except ImportError as exc:  # pragma: no cover - needs the agents extra
            raise DiscoveryAgentError(
                "deep discovery requires the agents extra: pip install 'adh[agents]'"
            ) from exc

        research = research_state if research_state is not None else DiscoveryResearchState()
        report_failures = 0

        def handle_report_error(
            error: Exception, *, kind: str = "schema", message: Any = None
        ) -> str:
            nonlocal report_failures
            report_failures += 1
            # LangChain wraps Pydantic's error in a ValueError. Keep field-level
            # feedback concise: raw/parsed telemetry already preserves the payload.
            source = getattr(error, "source", None)
            validation = getattr(source, "__cause__", None)
            if isinstance(validation, ValidationError):
                detail = "; ".join(
                    f"{'.'.join(map(str, item['loc'])) or 'report'}: {item['msg']}"
                    for item in validation.errors(include_input=False, include_url=False)[:5]
                )
            else:
                detail = _short(str(error), 1000)
            retry = report_failures <= REPORT_VALIDATION_RETRIES
            logger.warning(
                "deep-agent rejected DiscoveryReport (attempt %d, %s): %s",
                report_failures,
                "retrying" if retry else "stopping",
                detail,
            )
            if telemetry is not None:
                if message is None:
                    message = getattr(error, "ai_message", None)
                telemetry.emit(
                    "discovery",
                    "report.rejected",
                    status="failed",
                    parent_operation_id=parent_operation_id,
                    attributes={
                        "attempt": report_failures,
                        "kind": kind,
                        "retries_remaining": max(
                            0, REPORT_VALIDATION_RETRIES - report_failures + 1
                        ),
                        "action": "retry" if retry else "stop",
                        "error": detail,
                        "tool_call_ids": [
                            call.get("id")
                            for call in getattr(message, "tool_calls", [])
                            if call.get("name") == "DiscoveryReport"
                        ],
                    },
                )
            if not retry:
                raise DiscoveryAgentError(
                    f"deep agent produced {report_failures} invalid DiscoveryReport responses "
                    f"and exhausted {REPORT_VALIDATION_RETRIES} correction retries: {detail}"
                ) from error
            if kind == "completion":
                return (
                    f"DiscoveryReport rejected: {detail}\n\n"
                    f"Dataset goal: {goal_text(spec)}\n"
                    "Continue the existing research conversation, then call DiscoveryReport again."
                )
            return (
                f"DiscoveryReport rejected: {detail}. "
                "Call DiscoveryReport once with corrected arguments. Include separate top-level "
                "aspects and sources arrays of objects. Keep topic a short label; do not put "
                "serialized JSON or the source list inside a string. Reuse the existing research "
                "and candidates; correct the report without repeating completed searches. "
                "If no suitable external sources exist, explicitly pass sources=[] and explain "
                "the coverage gap in notes."
            )

        class DiscoveryGuardMiddleware(AgentMiddleware):
            def wrap_model_call(self, request, handler):
                if check_stop is not None:
                    check_stop()
                if hint := research.review_hint():
                    system = request.system_message
                    if system is None:
                        system = SystemMessage(content=hint)
                    else:
                        content = (
                            system.content + "\n\n" + hint
                            if isinstance(system.content, str)
                            else [*system.content, {"type": "text", "text": hint}]
                        )
                        system = system.model_copy(update={"content": content})
                    request = request.override(system_message=system)
                with capture_model_responses():
                    result = handler(request)
                if check_stop is not None:
                    check_stop()
                report = result.structured_response
                if isinstance(report, DiscoveryReport) and (
                    feedback := _empty_report_feedback(report, planning_gate, research)
                ):
                    # ToolStrategy has parsed valid JSON and added a success
                    # ToolMessage. Replace that acknowledgement before committing
                    # the graph state so this unfinished report cannot end the run.
                    from langchain_core.messages import AIMessage, ToolMessage

                    message = next(m for m in result.result if isinstance(m, AIMessage))
                    correction = handle_report_error(
                        ValueError(feedback), kind="completion", message=message
                    )
                    report_calls = {
                        call["id"]
                        for call in message.tool_calls
                        if call["name"] == "DiscoveryReport"
                    }
                    result = replace(
                        result,
                        structured_response=None,
                        result=[
                            (
                                m.model_copy(update={"content": correction, "status": "error"})
                                if isinstance(m, ToolMessage) and m.tool_call_id in report_calls
                                else m
                            )
                            for m in result.result
                        ],
                    )
                return result

            def wrap_tool_call(self, request, handler):
                if check_stop is not None:
                    check_stop()
                result = handler(request)
                if record_tool_result is not None:
                    record_tool_result(request.tool_call, result)
                if check_stop is not None:
                    check_stop()
                return result

        guards = [DiscoveryGuardMiddleware()]

        model = self.model
        if model is None or isinstance(model, str):
            from adh.llm import build_chat_model
            from adh.settings import get_settings

            model = build_chat_model(model or get_settings().discovery_model)

        if not install_response_logging(model):
            logger.debug(
                "raw HTTP response capture unavailable for %s; parsed responses will be logged",
                type(model).__name__,
            )

        def budget_status() -> str:
            if self.usage is None:
                return ""
            used = self.usage.total.total_tokens
            marker = " — WRAP UP NOW and emit your DiscoveryReport." if used >= 0.7 * budget else ""
            return f"discovery token budget: {used}/{budget} used{marker}"

        planning_gate = DiscoveryPlanningGate(frozenset(spec.discovery.channels))
        tools = build_discovery_tools(
            self.search,
            self.registry,
            work_dir,
            limit=self.config.max_candidates_per_aspect,
            budget_status=budget_status if budget else None,
            allowed_channels=planning_gate.allowed_channels,
            planning_gate=planning_gate,
            research_state=research,
        )
        tool_by_name = {tool.__name__: tool for tool in tools}
        # Custom middleware is not inherited by the SDK's default subagent.
        # Supply its base spec explicitly so delegated loops share our guards.
        subagents = [{**GENERAL_PURPOSE_SUBAGENT, "tools": tools, "middleware": guards}]
        if web_search := tool_by_name.get("web_search"):
            subagents.append(
                {
                    "name": "web-scout",
                    "description": (
                        "Focused web reconnaissance for one aspect: runs diverse web searches "
                        "and reports the most promising candidate pages."
                    ),
                    "system_prompt": DISCOVERY_WEB_SCOUT,
                    "tools": [web_search],
                    "middleware": guards,
                }
            )
        backend = FilesystemBackend(root_dir=work_dir, virtual_mode=True)
        return (
            create_deep_agent(
                model=model,
                tools=tools,
                system_prompt=DISCOVERY_AGENT_SYSTEM,
                middleware=guards,
                subagents=subagents,
                response_format=ToolStrategy(DiscoveryReport, handle_errors=handle_report_error),
                backend=backend,
            ),
            planning_gate,
        )

    @staticmethod
    def _constraints_text(spec: JobSpec) -> str:
        c = spec.discovery.constraints
        parts = []
        if c.include_domains:
            parts.append("only domains: " + ", ".join(c.include_domains))
        if c.exclude_domains:
            parts.append("never domains: " + ", ".join(c.exclude_domains))
        if c.recency_after:
            parts.append(f"published after {c.recency_after}")
        return "; ".join(parts) or "none"

    # ------------------------------------------------------------- plan mapping
    def _to_plan(self, report: DiscoveryReport, spec: JobSpec, job_id: str) -> FetchPlan:
        notes = ["discovery: deepagents research loop"] + list(report.notes)
        aspects = [
            Aspect(
                aspect_id=f"a{i}",
                topic=a.topic,
                rationale=a.rationale,
                keywords=tuple(k.strip().lower() for k in a.keywords if k.strip()),
            )
            for i, a in enumerate(report.aspects)
        ]
        allowed = set(spec.discovery.channels)

        synthesis: list[Aspect] = []
        if SourceChannel.SYNTHESIS in allowed:
            for i in report.synthesis_aspect_indexes:
                if 0 <= i < len(aspects):
                    synthesis.append(aspects[i])
                    notes.append(
                        f"aspect {aspects[i].aspect_id!r} routed to synthesis (model-known)"
                    )

        ranked: list[RankedSource] = []
        for src in report.sources:
            ranked_source = self._to_ranked(src, aspects, allowed, spec, notes)
            if ranked_source is not None:
                ranked.append(ranked_source)

        ranked = dedupe_and_rank(ranked)
        selected: list[RankedSource] = []
        per_aspect: dict[str, int] = {}
        limit = max(self.config.max_candidates_per_aspect, 0)
        for source in ranked:
            aspect_id = source.candidate.aspect_id
            count = per_aspect.get(aspect_id, 0)
            if count >= limit:
                continue
            selected.append(source)
            per_aspect[aspect_id] = count + 1
        dropped = len(ranked) - len(selected)
        if dropped:
            notes.append(
                f"discovery candidate cap removed {dropped} lower-ranked source(s) "
                f"(maximum {limit} per aspect)"
            )

        return FetchPlan(
            job_id=job_id,
            sources=tuple(selected),
            synthesis_aspects=tuple(synthesis),
            aspects=tuple(aspects),
            notes=notes,
        )

    def _to_ranked(
        self,
        src: ReportSource,
        aspects: list[Aspect],
        allowed: set[SourceChannel],
        spec: JobSpec,
        notes: list[str],
    ) -> RankedSource | None:
        """Map one reported source to a RankedSource; None (with a note) if rejected."""
        try:
            channel = SourceChannel(src.channel)
        except ValueError:
            notes.append(f"skip {src.locator}: unknown channel {src.channel!r}")
            return None
        if channel not in allowed or channel == SourceChannel.SYNTHESIS:
            notes.append(f"skip {src.locator}: channel {channel.value!r} not allowed")
            return None
        if not 0 <= src.aspect_index < len(aspects):
            notes.append(f"skip {src.locator}: bad aspect index {src.aspect_index}")
            return None
        if channel == SourceChannel.WEB and not domain_allowed(src.locator, spec):
            notes.append(f"skip {src.locator}: domain not allowed by constraints")
            return None

        aspect = aspects[src.aspect_index]
        prefix = "ds" if channel == SourceChannel.DATASETS else "web"
        cand = CandidateSource(
            source_id=f"{aspect.aspect_id}:{prefix}:{src.locator}",
            channel=channel,
            locator=src.locator,
            title=src.title or src.locator,
            aspect_id=aspect.aspect_id,
            snippet=src.snippet,
            license=src.license,
            published=src.published,
            discovered_via=src.discovered_via or "deep-agent",
        )
        shape = min(max(src.shape_match, 0.0), 1.0)
        quality = min(max(src.quality, 0.0), 1.0)
        est_yield = max(int(src.est_yield), 0)
        lic_ok = license_ok(src.license, spec)
        fresh_ok = freshness_ok(src.published, spec)
        score = score_candidate(
            shape_match=shape,
            quality=quality,
            est_yield=est_yield,
            channel=channel,
            license_ok=lic_ok,
            freshness_ok=fresh_ok,
            config=self.config,
        )
        evaluation = SourceEvaluation(
            source_id=cand.source_id,
            shape_match=shape,
            quality=quality,
            license_ok=lic_ok,
            freshness_ok=fresh_ok,
            estimated_yield=est_yield,
            score=round(score, 4),
            notes=src.notes,
        )
        return RankedSource(candidate=cand, evaluation=evaluation)


# ------------------------------------------------------------------- fallback
@dataclass
class FallbackDiscoveryAgent:
    """Primary discovery agent with a deterministic safety net (decision 3).

    Any primary failure — missing API key, transport error, recursion cap, junk
    structured output — degrades to the heuristic :class:`SourceDiscoveryAgent`
    instead of failing the run; the produced plan says so in its notes.
    """

    primary: DiscoveryAgent
    fallback: DiscoveryAgent = field(default_factory=SourceDiscoveryAgent)

    def discover(self, spec: JobSpec, work_dir: Path) -> FetchPlan:
        try:
            return self.primary.discover(spec, work_dir=work_dir)
        except Exception as exc:  # noqa: BLE001 - the fallback is the contract
            logger.warning("deep discovery failed (%s); using heuristic agent", exc)
            plan = self.fallback.discover(spec, work_dir=work_dir)
            plan.notes.append(f"deep discovery failed ({exc}); heuristic fallback plan")
            return plan
