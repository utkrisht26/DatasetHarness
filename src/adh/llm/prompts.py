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
