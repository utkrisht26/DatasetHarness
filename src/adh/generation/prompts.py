"""Generation prompt templates for the LLM generator (OUTLINE §2.2).

Prompts ask the model for a single JSON object so the output parses deterministically.
The shape depends on the verifiability tier: Tier-1 tasks must return their own
oracle (``tests`` for code, ``reference`` for math/short-answer) so the deterministic
verifier can decide correctness; Tier-2 tasks return only the ``input``/``output``
pair the LLM-jury will grade.
"""

from __future__ import annotations

from adh.contract import JobSpec, VerifiabilityTier

from .models import SeedPoint

_SYSTEM = (
    "You are a dataset generation engine. You produce ONE high-quality training record "
    "as a single minified JSON object and nothing else — no prose, no markdown fences. "
    "Every record must be standalone: it teaches the topic itself to a reader who will "
    "never see your reference material, so it never mentions a context, source, "
    "excerpt, page, or document. Use reference material only to keep facts accurate; "
    "never invent sources."
)

_GROUNDING_RULES = (
    "Use the reference material above only to make the record accurate. Write a "
    "standalone record about the Topic itself — the question must make sense to "
    "someone who has never seen this material, and neither field may mention it "
    '("the provided index", "this excerpt", "the source", ...). If the material '
    "contains no substantive information about the Topic (e.g. a navigation page, "
    'job listing, or unrelated subject), return {"skip": <one-line reason>} instead.'
)

_GROUNDING_BUDGET = 4000  # chars of retrieved grounding text to include


def system_prompt() -> str:
    return _SYSTEM


def _shape_instructions(tier: VerifiabilityTier, modality: str) -> str:
    if tier == VerifiabilityTier.HARD:
        if modality == "code":
            return (
                'Return JSON {"input": <problem statement>, "output": <complete Python '
                'solution>, "tests": <pytest asserts that import names from the solution '
                "and fail on a wrong answer>}. The tests MUST pass on your own output."
            )
        return (
            'Return JSON {"input": <problem>, "output": <full worked solution>, '
            '"reference": <the final answer only, e.g. a number or expression>}.'
        )
    # Tier 2 / 3 graded dimension
    return 'Return JSON {"input": <a question or instruction>, "output": <the ideal answer>}.'


def generation_prompt(seed: SeedPoint, spec: JobSpec, *, grounding: str = "") -> str:
    """Render the human prompt for one seed point."""
    tier = spec.verifiability_tier
    modality = spec.modality
    lines = [
        f"Topic: {seed.topic}",
        f"Keywords: {', '.join(seed.keywords) if seed.keywords else '(none)'}",
        f"Write for: {seed.persona}. Difficulty: {seed.difficulty}.",
        f"Task categories: {', '.join(spec.task_categories)}.",
        f"Language: {spec.languages[0] if spec.languages else 'en'}.",
    ]
    if grounding.strip():
        lines.append("\nReference material:\n---\n" + grounding[:_GROUNDING_BUDGET] + "\n---\n")
        lines.append(_GROUNDING_RULES)
    else:
        lines.append("\nNo source context is provided; generate from domain knowledge.")
    lines.append("\n" + _shape_instructions(tier, modality))
    return "\n".join(lines)
