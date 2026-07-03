"""Shared, provider-generic chat-model construction.

Any module that needs a live model (the web channel's relevance judge, an LLM-backed
discovery provider, Engine B judges, …) builds it here, so provider selection and
configuration live in exactly one place. The factory is a thin wrapper over LangChain's
``init_chat_model``, which accepts ``"<provider>:<model>"`` ids
(``"anthropic:claude-haiku-4-5"``, ``"openai:gpt-4o-mini"``, …).

LangChain is the optional ``agents`` extra, so it is imported lazily: the core stays
importable without it, and callers that never build a model never pay for it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    from langchain_core.language_models.chat_models import BaseChatModel

# Cheap, capable default for relevance/judging work; override per call site.
DEFAULT_CHAT_MODEL = "anthropic:claude-haiku-4-5"


def build_chat_model(model_id: str = DEFAULT_CHAT_MODEL, **kwargs: Any) -> "BaseChatModel":
    """Return a configured LangChain chat model for ``model_id``.

    ``model_id`` is a LangChain provider id (``"anthropic:…"``, ``"openai:…"``, …).
    Extra ``kwargs`` (``temperature``, ``max_tokens``, …) pass through to the provider.
    Raises a clear error if the ``agents`` extra (LangChain) is not installed.
    """
    try:
        from langchain.chat_models import init_chat_model
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise RuntimeError(
            "Building a chat model requires LangChain; install with: pip install 'adh[agents]'"
        ) from exc
    return init_chat_model(model_id, **kwargs)
