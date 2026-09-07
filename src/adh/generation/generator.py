"""Record generation (OUTLINE §2.2) — turn a grounded seed into candidate records.

The :class:`Generator` port has two implementations, mirroring the discovery layer:

* :class:`LlmGenerator` — the production path. Retrieves the seed's full grounding
  text from the corpus store, prompts a live model (via the shared ``build_chat_model``
  seam), parses the structured JSON payload, and meters every call into the run's
  :class:`~adh.llm.UsageTracker`.
* :class:`HeuristicGenerator` — the offline/test default. Deterministic and
  network-free: it emits genuinely verifier-passing Tier-1 pairs and grounding-based
  Tier-2 answers so the whole pipeline runs and is tested without a model or API key.

:func:`build_generator` picks the live path when the job configures a model and the
``agents`` extra is importable, and falls back to the heuristic otherwise.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from adh.contract import JobSpec, VerifiabilityTier, is_media_modality
from adh.corpus import CorpusStore
from adh.llm import UsageTracker
from adh.provenance import approx_token_count, sha256_prefixed

from .models import Candidate, SeedPoint, make_candidate_id
from .prompts import generation_prompt, system_prompt

logger = logging.getLogger(__name__)


@runtime_checkable
class Generator(Protocol):
    def generate(
        self, seed: SeedPoint, *, spec: JobSpec, store: CorpusStore | None = None
    ) -> list[Candidate]:
        """Produce zero or more candidate records for one seed point."""
        ...


# --------------------------------------------------------------------- helpers
def _task_category(spec: JobSpec) -> str | None:
    return spec.task_categories[0] if spec.task_categories else None


def _grounding_text(seed: SeedPoint, store: CorpusStore | None) -> str:
    """Retrieve the seed's full grounding text (context-offload read), else preview."""
    if store is not None and seed.grounding_uri:
        try:
            if store.exists(seed.grounding_uri):
                return store.get(seed.grounding_uri)
        except OSError as exc:  # pragma: no cover - store read failure is rare
            logger.warning("grounding read failed for %s: %s", seed.grounding_uri, exc)
    return seed.grounding_preview


def _source_url(seed: SeedPoint) -> str | None:
    url = seed.source_url
    return (
        url
        if isinstance(url, str) and url.startswith(("http://", "https://"))
        else None
    )


# --------------------------------------------------------------------- heuristic
@dataclass
class HeuristicGenerator:
    """Deterministic, offline generator — the test seam (no model, no network)."""

    def generate(
        self, seed: SeedPoint, *, spec: JobSpec, store: CorpusStore | None = None
    ) -> list[Candidate]:
        grounding = _grounding_text(seed, store)
        tier = spec.verifiability_tier
        if tier == VerifiabilityTier.HARD:
            payload = self._hard_payload(seed, spec)
        else:
            payload = self._soft_payload(seed, grounding)
        input_text, output_text, reference, tests = payload
        return [
            Candidate(
                candidate_id=make_candidate_id(
                    seed.job_id, seed.aspect_id, input_text, output_text
                ),
                job_id=seed.job_id,
                aspect_id=seed.aspect_id,
                verifiability_tier=tier,
                input=input_text,
                output=output_text,
                reference=reference,
                tests=tests,
                source_channel=seed.source_channel,
                source_url=_source_url(seed),
                source_license=seed.source_license,
                source_sha256=seed.source_sha256,
                derived_from=seed.derived_from,
                model_id="heuristic:adh",
                prompt_hash=sha256_prefixed(
                    f"{seed.seed_id}:{seed.persona}:{seed.difficulty}"
                ),
                generation_params=dict(spec.generation_params),
                pipeline_step="text-generation",
                input_tokens=approx_token_count(input_text),
                output_tokens=approx_token_count(output_text),
                language=spec.languages[0] if spec.languages else "en",
                task_category=_task_category(spec),
                cost_usd=0.0,
            )
        ]

    def _hard_payload(self, seed: SeedPoint, spec: JobSpec):
        import keyword

        kws = set(seed.keywords)
        if spec.modality == "code" or (
            {"code", "macro", "async", "rust", "python"} & kws
        ):
            raw = re.sub(
                r"[^a-z0-9_]",
                "",
                (seed.keywords[0] if seed.keywords else "echo").lower(),
            )
            fn = (
                f"{raw}_fn"
                if (not raw or not raw.isidentifier() or keyword.iskeyword(raw))
                else raw
            )
            code = f"def {fn}(x):\n    return x\n"
            tests = f"def test_{fn}():\n    assert {fn}(1) == 1\n    assert {fn}('a') == 'a'\n"
            return (
                f"Write a Python function `{fn}` that returns its argument.",
                code,
                None,
                tests,
            )
        # math default: a deterministic, symbolically-checkable arithmetic fact
        a, b = (len(seed.topic) % 9) + 1, (len(seed.persona) % 9) + 1
        return (f"What is {a} + {b}?", f"The answer is {a + b}.", str(a + b), None)

    def _soft_payload(self, seed: SeedPoint, grounding: str):
        excerpt = " ".join(grounding.split())[:400] if grounding else ""
        prefix = f"As {seed.persona}, " if seed.persona else ""
        question = f"{prefix}explain {seed.topic} at a {seed.difficulty} level."
        if excerpt:
            answer = f"{seed.topic.capitalize()}: {excerpt}"
        else:
            answer = (
                f"{seed.topic.capitalize()} concerns {', '.join(seed.keywords[:3]) or 'the topic'}. "
                f"At a {seed.difficulty} level, the key idea is to apply it deliberately."
            )
        return (question, answer, None, None)


# --------------------------------------------------------------------- llm
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

_CORRECTION = (
    "Your previous reply was not valid JSON. Respond again with ONLY the JSON "
    "object — no prose, no markdown fences."
)


@dataclass
class LlmGenerator:
    """Live generator over a real chat model (the ``agents`` extra)."""

    model_id: str
    usage: UsageTracker | None = None
    max_grounding_chars: int = 4000

    def __post_init__(self) -> None:
        from adh.llm import build_chat_model

        # Built eagerly so build_generator's fallback trips here (missing extra/key)
        # rather than mid-run; per-call params are applied at invoke time.
        self._model = build_chat_model(self.model_id)

    def generate(
        self, seed: SeedPoint, *, spec: JobSpec, store: CorpusStore | None = None
    ) -> list[Candidate]:
        from langchain_core.messages import HumanMessage, SystemMessage

        grounding = _grounding_text(seed, store)[: self.max_grounding_chars]
        sys = system_prompt()
        human = generation_prompt(seed, spec, grounding=grounding)
        gen_params = _invoke_params(spec.generation_params)
        response = self._model.invoke(
            [SystemMessage(content=sys), HumanMessage(content=human)], **gen_params
        )
        delta = self.usage.add_message(response) if self.usage is not None else None
        text = _text(response)
        payload = _parse_payload(text)
        if payload is None:
            # One bounded corrective re-ask: providers intermittently wrap or drop the
            # JSON despite the system prompt (b6b176e6 lost ~a third of its calls this
            # way). Still unparseable -> return empty and let routing escalate.
            payload, delta_re = self._reask(
                sys=sys, human=human, prior=text, gen_params=gen_params
            )
            if delta is not None and delta_re is not None:
                delta = delta + delta_re
        if payload is None:
            logger.warning(
                "generator produced unparseable output for seed %s — rejected reply head: %.200r",
                seed.seed_id,
                text[:200],
            )
            return []
        input_text = str(payload.get("input", "")).strip()
        output_text = str(payload.get("output", "")).strip()
        if not input_text or not output_text:
            return []
        return [
            Candidate(
                candidate_id=make_candidate_id(
                    seed.job_id, seed.aspect_id, input_text, output_text
                ),
                job_id=seed.job_id,
                aspect_id=seed.aspect_id,
                verifiability_tier=spec.verifiability_tier,
                input=input_text,
                output=output_text,
                reference=_opt_str(payload.get("reference")),
                tests=_opt_str(payload.get("tests")),
                source_channel=seed.source_channel,
                source_url=_source_url(seed),
                source_license=seed.source_license,
                source_sha256=seed.source_sha256,
                derived_from=seed.derived_from,
                model_id=self.model_id,
                prompt_hash=sha256_prefixed(human),
                system_prompt_hash=sha256_prefixed(sys),
                generation_params=dict(spec.generation_params),
                pipeline_step="text-generation",
                input_tokens=delta.input_tokens if delta else approx_token_count(human),
                output_tokens=(
                    delta.output_tokens if delta else approx_token_count(output_text)
                ),
                language=spec.languages[0] if spec.languages else "en",
                task_category=_task_category(spec),
                cost_usd=(delta.cost_usd or 0.0) if delta else 0.0,
            )
        ]

    def _reask(self, *, sys: str, human: str, prior: str, gen_params: dict):
        """Re-ask once with the rejected reply in context; returns (payload, delta)."""
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        response = self._model.invoke(
            [
                SystemMessage(content=sys),
                HumanMessage(content=human),
                AIMessage(content=prior),
                HumanMessage(content=_CORRECTION),
            ],
            **gen_params,
        )
        delta = self.usage.add_message(response) if self.usage is not None else None
        return _parse_payload(_text(response)), delta


def _invoke_params(generation_params: dict) -> dict:
    """Pass through the model params the contract's generation_params carries."""
    allowed = ("temperature", "top_p", "max_tokens", "seed")
    return {k: generation_params[k] for k in allowed if k in generation_params}


def _text(response) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):  # some providers return content blocks
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


def _parse_payload(text: str) -> dict | None:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass
    m = _JSON_RE.search(text or "")
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _opt_str(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def build_generator(spec: JobSpec, *, usage: UsageTracker | None = None) -> Generator:
    """Live routed ladder when models are configured and importable; else heuristic.

    Model selection is configuration, not code (Engine-A parity): the
    ``ADH_GENERATOR_MODELS`` deployment override wins when set, else the spec's
    ``generator_models`` ladder applies. The resolved list becomes a cheap->frontier
    :class:`~adh.generation.routing.RoutedGenerator`; the first tier is probed
    eagerly so the deterministic fallback still trips before any run starts.

    Media jobs (any acquired-not-synthesized modality) bypass the model ladder
    entirely: their payload is acquired, not synthesized, so a deterministic
    packager produces the candidates.
    """
    if is_media_modality(spec.modality):
        logger.info(
            "%s modality: using deterministic MediaPackager (no LLM calls)",
            spec.modality,
        )
        from .media_packager import MediaPackager

        return MediaPackager()

    from .routing import RoutedGenerator, routed_from_models

    models = _configured_generator_models(spec)
    if not models:
        return HeuristicGenerator()
    try:
        tiers = routed_from_models(models, usage=usage)
        tiers[0]()  # probe: missing extra / bad id fails here, not mid-run
    except (
        Exception
    ) as exc:  # noqa: BLE001 - no extra / no key -> deterministic fallback
        logger.warning("LlmGenerator unavailable (%s); using HeuristicGenerator", exc)
        return HeuristicGenerator()
    return RoutedGenerator(tiers=tiers)


def _configured_generator_models(spec: JobSpec) -> list[str]:
    """Deployment ladder (``ADH_GENERATOR_MODELS``) first, else the spec's own."""
    from adh.llm import GENERATOR_MODELS_ENV, env_model_list

    env_ids = env_model_list(GENERATOR_MODELS_ENV)
    if env_ids:
        logger.warning(
            "generator ladder overridden by %s=%s (spec configured %s)",
            GENERATOR_MODELS_ENV,
            ",".join(env_ids),
            list(spec.generator_models or []),
        )
        return env_ids
    return list(spec.generator_models or [])
