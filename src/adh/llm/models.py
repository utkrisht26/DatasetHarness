"""Shared, provider-generic chat-model construction.

Any module that needs a live model (the web channel's relevance judge, an LLM-backed
discovery provider, Engine B judges, …) builds it here, so provider selection and
configuration live in exactly one place. The factory is a thin wrapper over LangChain's
``init_chat_model``, which accepts ``"<provider>:<model>"`` ids
(``"anthropic:claude-haiku-4-5"``, ``"openai:gpt-4o-mini"``, …).

Where traffic goes is configuration, not code: several gateways expose
OpenAI-compatible APIs (OpenRouter, Together, Groq, vLLM, Ollama, …), so the
endpoint is chosen per environment instead of being hard-wired:

* ``ADH_LLM_BASE_URL`` — route every factory-built model through this endpoint.
* ``ADH_LLM_API_KEY``  — credential paired with that endpoint; without it the
  ambient provider variables (e.g. ``OPENAI_API_KEY``) still resolve.
* ``ADH_GENERATOR_MODELS`` — comma-separated cheap→frontier generation ladder;
  overrides ``generator_models`` in every job spec (Engine B).
* ``ADH_JUDGE_MODELS`` — comma-separated PoLL judge panel; overrides
  ``judge_config.models`` in every job spec (Engine B).

An explicit ``base_url`` / ``api_key`` argument always beats both, so a caller
can target any provider directly regardless of the environment. The model-list
variables are deployment-level redirects: when set they win over specs (loudly,
via logs — records still carry the model ids that actually ran), mirroring how
the discovery side takes its model from launch configuration.

LangChain is the optional ``agents`` extra, so it is imported lazily: the core stays
importable without it, and callers that never build a model never pay for it.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.language_models.chat_models import BaseChatModel

# Environment names for endpoint routing (see module docstring).
BASE_URL_ENV = "ADH_LLM_BASE_URL"
API_KEY_ENV = "ADH_LLM_API_KEY"
# Environment names for Engine B model selection — the same "configuration, not
# code" seam as the endpoint above (see module docstring).
GENERATOR_MODELS_ENV = "ADH_GENERATOR_MODELS"
JUDGE_MODELS_ENV = "ADH_JUDGE_MODELS"

# Cheap, capable default for relevance/judging work; override per call site.
DEFAULT_CHAT_MODEL = "openai:stealth/ox-alpha"


def env_model_list(env_name: str) -> list[str]:
    """Parse a comma-separated model-id list from ``env_name`` ([] when unset).

    Shared syntax for both Engine B seams:
    ``ADH_GENERATOR_MODELS="openai:gpt-4o-mini, anthropic:claude-opus-4-5"``
    configures a two-tier ladder; whitespace around ids is tolerated and empty
    entries are dropped.
    """
    raw = os.environ.get(env_name)
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def build_chat_model(model_id: str = DEFAULT_CHAT_MODEL, **kwargs: Any) -> "BaseChatModel":
    """Return a configured LangChain chat model for ``model_id``.

    ``model_id`` is a LangChain provider id (``"anthropic:…"``, ``"openai:…"``, …).
    Extra ``kwargs`` (``temperature``, ``max_tokens``, …) pass through to the provider.
    Endpoint routing follows the precedence in the module docstring: an explicit
    ``base_url``, else ``ADH_LLM_BASE_URL`` (with ``ADH_LLM_API_KEY`` paired along),
    else the provider's own default.
    Raises a clear error if the ``agents`` extra (LangChain) is not installed.
    """
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Building a chat model requires LangChain; install with: pip install 'adh[agents]'"
        ) from exc

    if "base_url" not in kwargs:
        base_url = os.environ.get(BASE_URL_ENV)
        if base_url:
            kwargs["base_url"] = base_url
            api_key = os.environ.get(API_KEY_ENV)
            if api_key and "api_key" not in kwargs:
                kwargs["api_key"] = api_key

    return init_chat_model(model_id, **kwargs)
