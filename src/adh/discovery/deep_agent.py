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

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, Field

from adh.contract import DiscoveryMode, JobSpec, SourceChannel
from adh.llm import TokenUsage, UsageTracker
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

logger = logging.getLogger(__name__)

# Generous headroom for decompose -> per-aspect search -> sample -> judge, but a
# hard stop for a model stuck re-calling tools (plan §failure-modes).
DEFAULT_RECURSION_LIMIT = 1500

# An upstream provider can return a well-formed-but-EMPTY completion (HTTP 200,
# no text, no tool calls, not even usage metadata — observed live via OpenRouter
# routing). That is a transient upstream fault, not a verdict on the task, so the
# whole research invocation is retried this many times before giving up.
EMPTY_COMPLETION_RETRIES = 2


class DiscoveryAgentError(RuntimeError):
    """The deep agent could not produce a structured discovery report."""


class DiscoveryBudgetExceeded(RuntimeError):
    """The discovery token budget was exhausted before a report was produced.

    Raised between model turns (never mid-call), so every token already spent is
    metered and folded into the run's :class:`UsageTracker`.
    """


# ------------------------------------------------------- the agent's wire format
class ReportAspect(BaseModel):
    """One sub-topic of the goal, as reported by the agent."""

    topic: str
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
        "", description="Keyword-rich summary of what lives there; steers the crawl frontier"
    )
    license: str | None = Field(None, description="License if visible (e.g. 'MIT'), else null")
    published: str | None = Field(None, description="Publish/update date if visible (YYYY-MM-DD)")
    discovered_via: str = Field(
        "", description="The search query that surfaced it, or 'model-knowledge' for proposed seeds"
    )
    shape_match: float = Field(0.0, description="0-1: does it contain records of the target shape?")
    quality: float = Field(0.0, description="0-1: signal vs noise of the content")
    est_yield: int = Field(0, description="Usable records this source could plausibly produce")
    notes: str = Field("", description="One-line judgement rationale")


class DiscoveryReport(BaseModel):
    """The deep agent's final structured output — everything a FetchPlan needs."""

    aspects: list[ReportAspect] = Field(default_factory=list)
    sources: list[ReportSource] = Field(default_factory=list)
    synthesis_aspect_indexes: list[int] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ------------------------------------------------------------- tools over ports
def build_discovery_tools(
    search: SearchProvider,
    registry: DatasetRegistry,
    work_dir: Path,
    *,
    limit: int = 8,
    budget_status: Callable[[], str] | None = None,
) -> list:
    """Wrap the existing search/registry ports as agent tools (no MCP).

    ``budget_status`` optionally supplies a one-line spend summary appended to
    every tool result — the soft nudge that lets the model wrap up on its own
    before the hard :class:`DiscoveryBudgetExceeded` guard fires.
    """

    counters = {"web": 0, "dataset": 0}

    def _nudge(text: str) -> str:
        if budget_status is None:
            return text
        line = budget_status()
        return f"{text}\n\n[budget] {line}" if line else text

    def web_search(query: str) -> str:
        """Search the web for candidate pages. Returns a virtual file path reference."""
        results = [
            {
                "url": hit.url,
                "title": hit.title,
                "snippet": hit.snippet,
                "license_hint": hit.license_hint,
                "published": hit.published,
            }
            for hit in search.search(query, limit=limit)
        ]
        counters["web"] += 1
        filename = f"web_search_{counters['web']}.json"
        rel_path = f"research/{filename}"
        abs_dir = Path(work_dir) / "research"
        abs_dir.mkdir(parents=True, exist_ok=True)
        abs_path = abs_dir / filename

        import json

        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump({"query": query, "results": results}, f, indent=2)

        return _nudge(
            f"Search results for query '{query}' (found {len(results)} results) "
            f"have been saved to virtual path '{rel_path}'. "
            f"You MUST use the `read_file` tool to read '{rel_path}' sequentially "
            f"to identify candidates (including locators, titles, snippets, and licenses)."
        )

    def dataset_search(query: str) -> str:
        """Search dataset registries (e.g. HF Hub) for existing datasets matching the query."""
        results = [
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
        counters["dataset"] += 1
        filename = f"dataset_search_{counters['dataset']}.json"
        rel_path = f"research/{filename}"
        abs_dir = Path(work_dir) / "research"
        abs_dir.mkdir(parents=True, exist_ok=True)
        abs_path = abs_dir / filename

        import json

        with open(abs_path, "w", encoding="utf-8") as f:
            json.dump({"query": query, "results": results}, f, indent=2)

        return _nudge(
            f"Dataset search results for query '{query}' (found {len(results)} results) "
            f"have been saved to virtual path '{rel_path}'. "
            f"You MUST use the `read_file` tool to read '{rel_path}' sequentially "
            f"to identify candidates."
        )

    def sample_dataset(dataset_id: str) -> str:
        """Fetch a small content sample from a dataset to judge shape/quality (empty if unavailable)."""
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

    return [web_search, dataset_search, sample_dataset]


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


def _build_tracer(tracker: UsageTracker | None, budget: int | None = None):
    """A LangChain callback that meters tokens, traces every step, AND enforces budget.

    Three jobs in one handler so there is a single ``on_llm_end`` pass:

    * **Metering** (always, when a ``tracker`` is given): folds each reply's
      usage — including provider-surfaced cost — into the run's
      :class:`UsageTracker`, so spend is captured even when the loop later errors
      or hits the recursion cap (decision 4: cost is real or unknown).
    * **Budget enforcement** (when a ``budget`` is given): right after folding a
      turn whose cumulative spend reaches the ceiling, raises
      :class:`DiscoveryBudgetExceeded` — the loop stops at spend, not only at
      the recursion limit, and every token already spent stays metered.
      ``raise_error = True`` makes handler exceptions propagate instead of being
      swallowed by the callback manager.
    * **Tracing** (always): logs each tool call, model turn with a running token
      total, and subagent step to the ``adh.discovery`` logger. These are INFO/
      DEBUG, so they surface only when logging is configured (``adh --verbose`` /
      ``adh serve``) and cost nothing otherwise. This is what turns "40k tokens
      then an error with no trace" into a step-by-step record of what ran.

    The instance carries live ``turns`` / ``tool_calls`` counters the caller reads
    for the start/finish/error summary lines.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    class _AgentTracer(BaseCallbackHandler):
        # propagate handler exceptions (the budget guard) out of the model call
        raise_error = True

        def __init__(self) -> None:
            self.turns = 0
            self.tool_calls = 0

        def on_tool_start(self, serialized, input_str, **kwargs) -> None:
            self.tool_calls += 1
            name = (serialized or {}).get("name") or kwargs.get("name") or "tool"
            logger.info("deep-agent → tool %s(%s)", name, _short(input_str))

        def on_tool_end(self, output, **kwargs) -> None:
            logger.debug("deep-agent ← %s", _short(output))

        def on_tool_error(self, error, **kwargs) -> None:
            logger.warning("deep-agent ✗ tool error: %s", _short(error))

        def on_llm_end(self, response, **kwargs) -> None:
            delta = 0
            for generations in response.generations:
                for generation in generations:
                    message = getattr(generation, "message", None)
                    if message is None:
                        continue
                    self.turns += 1
                    usage = (
                        tracker.add_message(message)
                        if tracker is not None
                        else TokenUsage.from_message(message)
                    )
                    delta += usage.total_tokens
            total = tracker.total.total_tokens if tracker is not None else delta
            logger.info("deep-agent · model turn %d (+%d tok, %d total)", self.turns, delta, total)
            if budget is not None and tracker is not None and total >= budget:
                raise DiscoveryBudgetExceeded(
                    f"discovery token budget exhausted: {total}/{budget} tokens used"
                )

        def on_llm_error(self, error, **kwargs) -> None:
            logger.warning("deep-agent ✗ model error: %s", _short(error))

    return _AgentTracer()


# ------------------------------------------------------------------- the agent
@dataclass
class DeepAgentDiscoveryAgent:
    """Discovery driven by a deepagents research loop over the existing ports.

    ``model`` is a LangChain chat model or a ``"<provider>:<model>"`` id (``None``
    uses :data:`adh.llm.DEFAULT_CHAT_MODEL`). Everything is injectable for offline
    tests: the in-memory ports feed the tools, and a scripted fake chat model
    drives the very same ``create_deep_agent`` loop with no network.
    """

    model: Any = None
    search: SearchProvider = field(default_factory=InMemorySearchProvider)
    registry: DatasetRegistry = field(default_factory=InMemoryDatasetRegistry)
    config: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    usage: UsageTracker | None = None
    recursion_limit: int = DEFAULT_RECURSION_LIMIT
    token_budget: int | None = None  # None -> derive from spec.limits.max_llm_tokens

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
        budget = self._effective_budget(spec)
        graph = self._build_graph(work_dir, budget=budget)
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
        tracer = _build_tracer(self.usage, budget=budget)
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
                    config={"recursion_limit": self.recursion_limit, "callbacks": [tracer]},
                )
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

    def _build_graph(self, work_dir: Path, *, budget: int | None = None):
        try:
            from deepagents import create_deep_agent
            from deepagents.backends import FilesystemBackend
            from langchain.agents.structured_output import ToolStrategy
        except ImportError as exc:  # pragma: no cover - needs the agents extra
            raise DiscoveryAgentError(
                "deep discovery requires the agents extra: pip install 'adh[agents]'"
            ) from exc

        model = self.model
        if model is None or isinstance(model, str):
            from adh.llm import DEFAULT_CHAT_MODEL, build_chat_model

            model = build_chat_model(model or DEFAULT_CHAT_MODEL)

        def budget_status() -> str:
            if self.usage is None:
                return ""
            used = self.usage.total.total_tokens
            marker = " — WRAP UP NOW and emit your DiscoveryReport." if used >= 0.7 * budget else ""
            return f"discovery token budget: {used}/{budget} used{marker}"

        tools = build_discovery_tools(
            self.search,
            self.registry,
            work_dir,
            limit=self.config.max_candidates_per_aspect,
            budget_status=budget_status if budget else None,
        )
        web_scout = {
            "name": "web-scout",
            "description": (
                "Focused web reconnaissance for one aspect: runs diverse web searches "
                "and reports the most promising candidate pages."
            ),
            "system_prompt": DISCOVERY_WEB_SCOUT,
            "tools": [tools[0]],  # web_search only
        }
        backend = FilesystemBackend(root_dir=work_dir, virtual_mode=True)
        return create_deep_agent(
            model=model,
            tools=tools,
            system_prompt=DISCOVERY_AGENT_SYSTEM,
            subagents=[web_scout],
            response_format=ToolStrategy(DiscoveryReport),
            backend=backend,
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

        return FetchPlan(
            job_id=job_id,
            sources=tuple(dedupe_and_rank(ranked)),
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
