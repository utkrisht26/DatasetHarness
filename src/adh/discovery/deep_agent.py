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
from typing import Any

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
DEFAULT_RECURSION_LIMIT = 50


class DiscoveryAgentError(RuntimeError):
    """The deep agent could not produce a structured discovery report."""


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

    channel: str = Field("web", description="'web' (page to crawl) or 'datasets' (existing dataset)")
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
    search: SearchProvider, registry: DatasetRegistry, work_dir: Path, *, limit: int = 8
) -> list:
    """Wrap the existing search/registry ports as agent tools (no MCP)."""

    counters = {"web": 0, "dataset": 0}

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

        return (
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

        return (
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

        return (
            f"Sample content for dataset '{dataset_id}' has been saved to virtual path '{rel_path}'. "
            f"You MUST use the `read_file` tool to view the content of '{rel_path}'."
        )

    return [web_search, dataset_search, sample_dataset]


def _short(text: object, limit: int = 160) -> str:
    """One-line, length-capped view of a value for a log line."""
    s = " ".join(str(text).split())
    return s if len(s) <= limit else s[: limit - 1] + "…"


def _build_tracer(tracker: UsageTracker | None):
    """A LangChain callback that meters tokens AND traces every agent step.

    Two jobs in one handler so there is a single ``on_llm_end`` pass:

    * **Metering** (always, when a ``tracker`` is given): folds each reply's
      usage — including provider-surfaced cost — into the run's
      :class:`UsageTracker`, so spend is captured even when the loop later errors
      or hits the recursion cap (decision 4: cost is real or unknown).
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
            logger.info(
                "deep-agent · model turn %d (+%d tok, %d total)", self.turns, delta, total
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

    def discover(self, spec: JobSpec, work_dir: Path) -> FetchPlan:
        job_id = spec.job_id or spec.name
        if spec.discovery.mode == DiscoveryMode.FIXED:
            # Nothing to research: the user gave exact sources.
            return fixed_plan(spec, job_id, [])

        report = self._run_agent(spec, job_id, work_dir)
        return self._to_plan(report, spec, job_id)

    # ---------------------------------------------------------------- research
    def _run_agent(self, spec: JobSpec, job_id: str, work_dir: Path) -> DiscoveryReport:
        graph = self._build_graph(work_dir)
        task = DISCOVERY_AGENT_TASK.format(
            goal=goal_text(spec),
            target_shape=target_shape(spec),
            channels=", ".join(c.value for c in spec.discovery.channels),
            constraints=self._constraints_text(spec),
        )
        # One handler meters tokens/cost as each model call ends AND traces every
        # tool call / model turn, so the run is observable live and its spend is
        # captured even when the loop errors or hits the recursion cap.
        tracer = _build_tracer(self.usage)
        logger.info(
            "deep-agent starting: job=%s goal=%r channels=%s recursion_limit=%d",
            job_id,
            _short(goal_text(spec), 80),
            ",".join(c.value for c in spec.discovery.channels),
            self.recursion_limit,
        )
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
        spent = self.usage.total.total_tokens if self.usage is not None else 0
        if not isinstance(report, DiscoveryReport):
            logger.error(
                "deep-agent produced no structured report after %d turn(s), %d token(s)",
                tracer.turns,
                spent,
            )
            raise DiscoveryAgentError("deep agent produced no structured DiscoveryReport")
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

    def _build_graph(self, work_dir: Path):
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

        tools = build_discovery_tools(
            self.search, self.registry, work_dir, limit=self.config.max_candidates_per_aspect
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
                    notes.append(f"aspect {aspects[i].aspect_id!r} routed to synthesis (model-known)")

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
