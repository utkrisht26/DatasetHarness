"""Tests for the central, side-effect-free application settings contract."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from adh import cli
from adh.settings import DEFAULT_DISCOVERY_MODEL, Settings, get_settings


def test_empty_environment_uses_safe_defaults() -> None:
    settings = get_settings({})

    assert settings.discovery_model == DEFAULT_DISCOVERY_MODEL
    assert settings.log_level == "INFO"
    assert settings.generator_models == ()
    assert settings.judge_models == ()
    assert settings.cors_origins == ()
    assert settings.service_api_key_value is None


def test_environment_lists_urls_and_secrets_are_normalized() -> None:
    settings = Settings.from_env(
        {
            "ADH_DISCOVERY_MODEL": " anthropic:claude-haiku-4-5 ",
            "ADH_LLM_BASE_URL": "http://localhost:11434/v1/",
            "ADH_LLM_API_KEY": "gateway-secret",
            "ADH_GENERATOR_MODELS": " openai:cheap, anthropic:frontier ,,",
            "ADH_JUDGE_MODELS": "openai:judge-a,anthropic:judge-b",
            "ADH_CORS_ORIGINS": "http://localhost:3000/, https://console.example.com",
            "ADH_LOG_LEVEL": " debug ",
        }
    )

    assert settings.discovery_model == "anthropic:claude-haiku-4-5"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_api_key_value == "gateway-secret"
    assert "gateway-secret" not in repr(settings)
    assert settings.generator_models == ("openai:cheap", "anthropic:frontier")
    assert settings.judge_models == ("openai:judge-a", "anthropic:judge-b")
    assert settings.cors_origins == (
        "http://localhost:3000",
        "https://console.example.com",
    )
    assert settings.log_level == "DEBUG"


def test_pii_entities_are_parsed_and_upper_cased() -> None:
    assert get_settings({}).pii_entities == ()
    settings = Settings.from_env({"ADH_PII_ENTITIES": " person, email_address ,,"})
    assert settings.pii_entities == ("PERSON", "EMAIL_ADDRESS")


@pytest.mark.parametrize(
    "environment",
    [
        {"ADH_LLM_BASE_URL": "localhost:11434/v1"},
        {"ADH_LLM_BASE_URL": "ftp://models.example.com"},
        {"ADH_CORS_ORIGINS": "https://console.example.com/application"},
        {"ADH_CORS_ORIGINS": "*"},
        {"ADH_LOG_LEVEL": "verbose"},
    ],
)
def test_invalid_endpoint_configuration_fails_fast(environment: dict[str, str]) -> None:
    with pytest.raises(ValidationError):
        Settings.from_env(environment)


def test_discovery_model_is_independent_from_judge_panel() -> None:
    settings = Settings.from_env({"ADH_JUDGE_MODELS": "openai:a,anthropic:b"})

    assert settings.discovery_model == DEFAULT_DISCOVERY_MODEL
    assert settings.judge_models == ("openai:a", "anthropic:b")


def test_cli_uses_configured_discovery_model_for_every_discovery_command(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ADH_DISCOVERY_MODEL", "anthropic:configured")
    parser = cli.build_parser()

    for command in ("discover", "run", "loop", "serve"):
        args = parser.parse_args([command, "spec.yaml"] if command != "serve" else [command])
        assert args.discovery_model == "anthropic:configured"
