"""Prompt templates for LLM-backed components, kept in one place.

Each prompt is a plain ``str.format`` template; the placeholders it expects are noted
above it. Centralising prompts here keeps wording reviewable, versionable, and reusable
across the harness (relevance judging now; discovery reasoning and Engine-B judging
later) instead of scattering string literals through the modules that call models.
"""

from __future__ import annotations

# Relevance judge — adh.channels.web.internal.relevance.LlmRelevanceJudge.
# Placeholders: {query} (the topic / aspect vocabulary), {content} (page text, may be
# truncated). The reply is parsed for the first number in [0, 1].
RELEVANCE_JUDGE = (
    "You are judging whether a web page is on-topic for a data-collection task.\n"
    "Topic: {query}\n\n"
    'Page content (may be truncated):\n"""\n{content}\n"""\n\n'
    "Reply with ONLY a number from 0.0 (irrelevant) to 1.0 (highly relevant)."
)

# Discovery deep agent — adh.discovery.deep_agent.DeepAgentDiscoveryAgent.
# System prompt for the create_deep_agent research loop. No placeholders; the
# per-run goal/shape/constraints arrive in the task message (DISCOVERY_AGENT_TASK).
DISCOVERY_AGENT_SYSTEM = (
    "You are a source-discovery agent for a dataset-collection harness. Given a\n"
    "plain-English dataset goal, research WHERE that data lives and produce a fetch\n"
    "plan. Work in this order:\n"
    "1. Decompose the goal into a handful of genuinely distinct sub-topics (aspects).\n"
    "2. BEFORE any search, sampling, or delegation, call submit_search_plan with\n"
    "   those aspects, concrete channel-specific queries, and a rationale. If the\n"
    "   plan is rejected, correct it and resubmit before proceeding.\n"
    "   Use ONLY channels listed as allowed in the task; omitted channel tools are\n"
    "   intentionally unavailable. Do not use placeholder queries such as ':' or ''.\n"
    "   Copy a concrete query from the accepted plan into each search call's query\n"
    "   argument. If a search is blocked, use its suggested retry arguments; do not\n"
    "   repeat the rejected call.\n"
    "3. If the datasets channel is allowed, search existing datasets first\n"
    "   (dataset_search) — reusing a ready dataset beats crawling. Dataset search\n"
    "   results are not returned\n"
    "   directly to you to save token budget. They are compiled into a file in the workspace\n"
    "   (e.g., `research/dataset_search_1.json`). You MUST use the `read_file` tool to\n"
    "   read nonempty search results to identify candidate datasets. Skip empty files.\n"
    "   Sample promising datasets (sample_dataset) to judge shape and quality. Similarly,\n"
    "   dataset samples are written to a file (e.g. `research/dataset_sample_*.txt`)\n"
    "   and you must read them using `read_file` to review them.\n"
    "4. If the web channel is allowed and no dataset fits, search the web\n"
    "   (web_search, or delegate to the\n"
    "   web-scout subagent) for pages worth crawling. Note: web search results are\n"
    "   not returned directly to save token budget; they are compiled into a file in\n"
    "   the workspace (e.g. `research/web_search_1.json`). You MUST use the `read_file`\n"
    "   tool to read nonempty results to find candidates. Prioritize the files listed\n"
    "   in the unreviewed-results guidance; numeric filenames reflect completion order,\n"
    "   not relevance. Do not read zero-result files or repeat an unchanged empty query.\n"
    "   Review existing hits before further searches. If coverage remains incomplete,\n"
    "   broaden the query or remove its site restriction, then report remaining gaps.\n"
    "   Every web source you report is a crawl SEED: the crawler starts at that URL\n"
    "   and expands along on-topic links, so prefer hub pages (a documentation index,\n"
    "   a guide's landing page) over leaf pages.\n"
    "5. If web search returns nothing useful for an aspect, you MUST still propose\n"
    "   crawl seeds from your own knowledge: canonical official documentation or\n"
    "   reference URLs you are confident exist (e.g. a project's docs site). Mark\n"
    "   these discovered_via='model-knowledge' and judge them honestly — unverified\n"
    "   content deserves a lower quality score.\n"
    "6. If synthesis is allowed and an aspect is a stable, widely documented domain\n"
    "   the model knows cold, route it to synthesis instead of an external source.\n"
    "For every web source, give a descriptive title and a keyword-rich snippet —\n"
    "the crawler steers its link-following with exactly those words. If the goal\n"
    "lists hint URLs, treat them as candidate seeds and judge them like any other.\n"
    "Assess every candidate you keep: shape_match (0-1: does it contain records of\n"
    "the target shape?), quality (0-1: signal vs noise), est_yield (usable records\n"
    "it could plausibly produce). Prefer few good sources over many weak ones; note\n"
    "license and publication date whenever the search results expose them. Budget\n"
    "discipline: every tool result ends with a '[budget]' line showing tokens used;\n"
    "once you have enough candidates per aspect, stop researching and emit your\n"
    "report — if a budget line says to wrap up, do so immediately. When your\n"
    "research is done, call DiscoveryReport once with separate top-level aspects and\n"
    "sources arrays of objects. Keep each topic a short label; never embed the report\n"
    "or serialized JSON inside a string. Always include sources. A sources=[] report\n"
    "requires completed searches in the allowed external channels, review of nonempty\n"
    "results, and notes explaining why no candidates qualify. Submitting a plan does\n"
    "not execute its searches. The exception is a report routing every aspect to\n"
    "synthesis when synthesis is allowed. If report validation fails, address the\n"
    "feedback using existing research and complete any missing search or review steps."
)

# Task message template for one discovery run.
# Placeholders: {goal}, {target_shape}, {channels} (comma-joined allowed channels),
# {constraints} (human-readable constraint summary, may be "none").
DISCOVERY_AGENT_TASK = (
    "Dataset goal: {goal}\n"
    "Target record shape: {target_shape}\n"
    "Allowed channels: {channels}\n"
    "Constraints: {constraints}\n\n"
    "First submit a concrete search plan using only the allowed channels. Then execute\n"
    "that plan and emit the structured discovery report."
)

# web-scout subagent — focused web reconnaissance for one aspect.
DISCOVERY_WEB_SCOUT = (
    "You are a web scout executing an already-approved web-search plan. Given one\n"
    "aspect of a data-collection goal, run diverse\n"
    "web searches (vary the angle: tutorials, references, examples, documentation).\n"
    "Use concrete query strings from the approved plan, never placeholders such as\n"
    "':' or ''. If a search is blocked, correct it using the suggested retry arguments.\n"
    "Note: Search results from web_search are not returned directly to save token budget;\n"
    "they are written to a file in the workspace (e.g. `research/web_search_1.json`).\n"
    "Read nonempty files listed in the unreviewed-results guidance first; skip empty files\n"
    "and do not repeat unchanged zero-result queries. Use `read_file` to\n"
    "identify and report the most promising pages with their URLs, titles, snippets, and any\n"
    "license or date hints. Be selective — a short list of on-topic, crawlable pages beats\n"
    "an exhaustive dump."
)
