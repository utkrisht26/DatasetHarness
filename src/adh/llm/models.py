"""Shared, provider-generic chat-model construction.

Any module that needs a live model (the web channel's relevance judge, an LLM-backed
discovery provider, Engine B judges, …) builds it here, so provider selection and
configuration live in exactly one place. The factory is a thin wrapper over LangChain's
``init_chat_model``, which accepts ``"<provider>:<model>"`` ids
(``"anthropic:claude-haiku-4-5"``, ``"openai:gpt-4o-mini"``, …).

Where traffic goes is configuration, not code: several gateways expose
OpenAI-compatible APIs (OpenRouter, Together, Groq, vLLM, Ollama, …), so the
endpoint is chosen per environment instead of being hard-wired:

* ``ADH_DISCOVERY_MODEL`` — default model for deep source discovery.
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

from typing import TYPE_CHECKING, Any

from adh.settings import (
    DEFAULT_DISCOVERY_MODEL,
    GENERATOR_MODELS_ENV as SETTINGS_GENERATOR_MODELS_ENV,
    JUDGE_MODELS_ENV as SETTINGS_JUDGE_MODELS_ENV,
    LLM_API_KEY_ENV,
    LLM_BASE_URL_ENV,
    env_model_list as settings_env_model_list,
    get_settings,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.language_models.chat_models import BaseChatModel

# Backward-compatible names; the canonical registry lives in ``adh.settings``.
BASE_URL_ENV = LLM_BASE_URL_ENV
API_KEY_ENV = LLM_API_KEY_ENV
GENERATOR_MODELS_ENV = SETTINGS_GENERATOR_MODELS_ENV
JUDGE_MODELS_ENV = SETTINGS_JUDGE_MODELS_ENV

# Backward-compatible default for call sites that do not accept a settings object.
DEFAULT_CHAT_MODEL = DEFAULT_DISCOVERY_MODEL
# Per-request ceiling. A stalled upstream response otherwise blocks a run forever
# (rust-async-live-small hung 38 min); generous because a reasoning model can
# legitimately think for minutes on a 32k-token completion.
DEFAULT_REQUEST_TIMEOUT_S = 600.0


def env_model_list(env_name: str) -> list[str]:
    """Backward-compatible model-list accessor backed by central settings."""
    return settings_env_model_list(env_name)


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

    kwargs.setdefault("timeout", DEFAULT_REQUEST_TIMEOUT_S)
    if "base_url" not in kwargs:
        settings = get_settings()
        if settings.llm_base_url:
            kwargs["base_url"] = settings.llm_base_url
            if settings.llm_api_key_value and "api_key" not in kwargs:
                kwargs["api_key"] = settings.llm_api_key_value

    return init_chat_model(model_id, **kwargs)
