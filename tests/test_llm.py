"""Tests for the shared LLM utilities: model factory, token usage, and prompts."""

from __future__ import annotations

from types import SimpleNamespace

import adh.llm
from adh.llm import DEFAULT_CHAT_MODEL, TokenUsage, UsageTracker, build_chat_model
from adh.llm.prompts import RELEVANCE_JUDGE


def test_default_model_is_a_provider_qualified_id():
    assert ":" in DEFAULT_CHAT_MODEL  # "<provider>:<model>" form for init_chat_model


def _fake_init(monkeypatch):
    """Patch langchain's init_chat_model and return the kwargs-seen dict."""
    seen = {}

    def fake_init(model_id, **kw):
        seen["model_id"], seen["kw"] = model_id, kw
        return object()

    # build_chat_model imports init_chat_model lazily from langchain.chat_models.
    import langchain.chat_models as lcm

    monkeypatch.setattr(lcm, "init_chat_model", fake_init)
    return seen


def test_build_chat_model_returns_an_invokable_model():
    # Construction must not require an API key (no network until .invoke()).
    model = build_chat_model("anthropic:claude-haiku-4-5")
    assert hasattr(model, "invoke")


def test_build_chat_model_injects_nothing_without_configuration(monkeypatch):
    # With no ADH_LLM_* configuration the factory is pure passthrough: several
    # OpenAI-compatible gateways exist, and which one serves traffic must come
    # from configuration, never be hard-wired in code.
    monkeypatch.delenv(adh.llm.models.BASE_URL_ENV, raising=False)
    monkeypatch.delenv(adh.llm.models.API_KEY_ENV, raising=False)
    seen = _fake_init(monkeypatch)

    build_chat_model("openai:gpt-4o-mini", temperature=0, max_tokens=16)
    assert seen["model_id"] == "openai:gpt-4o-mini"
    assert seen["kw"] == {"temperature": 0, "max_tokens": 16}


def test_build_chat_model_routes_through_env_base_url(monkeypatch):
    monkeypatch.setenv(adh.llm.models.BASE_URL_ENV, "https://api.together.xyz/v1")
    monkeypatch.delenv(adh.llm.models.API_KEY_ENV, raising=False)
    seen = _fake_init(monkeypatch)

    build_chat_model("openai:gpt-4o-mini", temperature=0)
    assert seen["kw"]["base_url"] == "https://api.together.xyz/v1"
    assert seen["kw"]["temperature"] == 0


def test_env_base_url_pairs_with_env_api_key(monkeypatch):
    monkeypatch.setenv(adh.llm.models.BASE_URL_ENV, "https://api.together.xyz/v1")
    monkeypatch.setenv(adh.llm.models.API_KEY_ENV, "test-key")
    seen = _fake_init(monkeypatch)

    build_chat_model("openai:gpt-4o-mini")
    assert seen["kw"]["base_url"] == "https://api.together.xyz/v1"
    assert seen["kw"]["api_key"] == "test-key"


def test_explicit_base_url_beats_env_and_stays_unpaired(monkeypatch):
    monkeypatch.setenv(adh.llm.models.BASE_URL_ENV, "https://api.together.xyz/v1")
    monkeypatch.setenv(adh.llm.models.API_KEY_ENV, "test-key")
    seen = _fake_init(monkeypatch)

    build_chat_model("openai:gpt-4o-mini", base_url="http://localhost:11434/v1")
    assert seen["kw"]["base_url"] == "http://localhost:11434/v1"
    assert "api_key" not in seen["kw"]  # endpoint-specific credentials stay the caller's job


def test_build_chat_model_is_exported_from_package():
    assert adh.llm.build_chat_model is build_chat_model


# --------------------------------------------------------------------- model-list env
def test_engine_b_model_list_env_names_are_stable():
    # Documented configuration surface (see adh.llm.models docstring) — renames
    # would silently break deployments, so pin them.
    assert adh.llm.models.GENERATOR_MODELS_ENV == "ADH_GENERATOR_MODELS"
    assert adh.llm.models.JUDGE_MODELS_ENV == "ADH_JUDGE_MODELS"


def test_env_model_list_parses_comma_separated_ids(monkeypatch):
    monkeypatch.setenv("ADH_TEST_MODELS", " openai:a , anthropic:b ,,")
    assert adh.llm.models.env_model_list("ADH_TEST_MODELS") == ["openai:a", "anthropic:b"]
    assert adh.llm.models.env_model_list("ADH_TEST_MISSING") == []
    monkeypatch.setenv("ADH_TEST_EMPTY", "  ,  ")
    assert adh.llm.models.env_model_list("ADH_TEST_EMPTY") == []


def test_env_model_list_is_exported_from_package():
    assert adh.llm.env_model_list is adh.llm.models.env_model_list


# --------------------------------------------------------------------- token usage
def test_token_usage_arithmetic():
    a = TokenUsage(input_tokens=10, output_tokens=2)
    b = TokenUsage(input_tokens=5, output_tokens=3)
    total = a + b
    assert (total.input_tokens, total.output_tokens, total.total_tokens) == (15, 5, 20)


def test_token_usage_from_message_usage_metadata():
    msg = SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 20})
    u = TokenUsage.from_message(msg)
    assert (u.input_tokens, u.output_tokens, u.total_tokens) == (100, 20, 120)


def test_token_usage_from_message_response_metadata_fallback():
    # OpenAI-style nested token_usage with no normalised usage_metadata.
    msg = SimpleNamespace(
        response_metadata={"token_usage": {"prompt_tokens": 7, "completion_tokens": 3}}
    )
    u = TokenUsage.from_message(msg)
    assert (u.input_tokens, u.output_tokens) == (7, 3)


def test_token_usage_from_message_handles_missing():
    assert TokenUsage.from_message(SimpleNamespace()).total_tokens == 0


def test_usage_tracker_accumulates():
    tracker = UsageTracker()
    tracker.add(TokenUsage(10, 2))
    delta = tracker.add_message(
        SimpleNamespace(usage_metadata={"input_tokens": 5, "output_tokens": 1})
    )
    assert delta.total_tokens == 6
    assert tracker.calls == 2
    assert tracker.total.total_tokens == 18
    assert tracker.to_dict() == {
        "input_tokens": 15,
        "output_tokens": 3,
        "total_tokens": 18,
        "cost_usd": None,  # no provider-surfaced cost -> reported unknown (decision 4)
        "calls": 2,
    }


# --------------------------------------------------------------------- prompts
def test_relevance_prompt_has_expected_placeholders():
    rendered = RELEVANCE_JUDGE.format(query="django views", content="a view handles a request")
    assert "django views" in rendered
    assert "a view handles a request" in rendered
