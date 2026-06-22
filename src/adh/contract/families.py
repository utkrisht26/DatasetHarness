"""Model-family detection for the PoLL (Panel-of-LLMs) judge invariant.

The contract requires Tier 2/3 jobs to use a judge panel drawn from ``>=2 distinct
model families`` (PHASE0-CONTRACT.md §0.3, RESEARCH.md §QC — disjoint-family panels
beat a single GPT-4 judge at lower cost and reduce self-preference bias).

A model id follows the ``provider:model`` convention used throughout the contract
(e.g. ``anthropic:claude-sonnet-4-6``, ``openai:gpt-judge``). The family is the
provider prefix. A bare id with no prefix is treated as its own family (its full
string), which is conservative: two distinct bare names still count as two families.
"""

from __future__ import annotations


def model_family(model_id: str) -> str:
    """Return the family key for a model id.

    ``"anthropic:claude-haiku-4-5"`` -> ``"anthropic"``.
    ``"gpt-judge"`` -> ``"gpt-judge"`` (no prefix: the whole id is its own family).
    """
    cleaned = model_id.strip()
    if ":" in cleaned:
        return cleaned.split(":", 1)[0].strip().lower()
    return cleaned.lower()


def distinct_families(model_ids: list[str]) -> set[str]:
    """The set of distinct families represented by ``model_ids``."""
    return {model_family(m) for m in model_ids if m and m.strip()}
