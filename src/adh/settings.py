"""Validated application settings sourced from environment variables.

This module is the single registry for ADH-owned environment variables. It does
not load ``.env`` or cache values at import time: library imports stay free of
side effects, while the CLI explicitly loads ``.env`` before parsing commands.
Provider-native variables such as ``OPENAI_API_KEY`` remain owned by their SDKs.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, SecretStr, field_validator

DISCOVERY_MODEL_ENV = "ADH_DISCOVERY_MODEL"
LLM_BASE_URL_ENV = "ADH_LLM_BASE_URL"
LLM_API_KEY_ENV = "ADH_LLM_API_KEY"
GENERATOR_MODELS_ENV = "ADH_GENERATOR_MODELS"
JUDGE_MODELS_ENV = "ADH_JUDGE_MODELS"
TAVILY_API_KEY_ENV = "TAVILY_API_KEY"
PEXELS_API_KEY_ENV = "PEXELS_API_KEY"
OPENVERSE_API_KEY_ENV = "OPENVERSE_API_KEY"
SERVICE_API_KEY_ENV = "ADH_API_KEY"
CORS_ORIGINS_ENV = "ADH_CORS_ORIGINS"
LOG_LEVEL_ENV = "ADH_LOG_LEVEL"

DEFAULT_DISCOVERY_MODEL = "openai:gpt-4o-mini"
DEFAULT_LOG_LEVEL = "INFO"
VALID_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})


def _optional(value: str | None) -> str | None:
    normalized = (value or "").strip()
    return normalized or None


def _csv(value: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (value or "").split(",") if part.strip())


def env_model_list(env_name: str, environ: Mapping[str, str] | None = None) -> list[str]:
    """Return a comma-separated model list from an environment mapping."""
    source = os.environ if environ is None else environ
    return list(_csv(source.get(env_name)))


class Settings(BaseModel):
    """Immutable, validated snapshot of ADH's deployment configuration."""

    model_config = ConfigDict(frozen=True)

    discovery_model: str = DEFAULT_DISCOVERY_MODEL
    llm_base_url: str | None = None
    llm_api_key: SecretStr | None = None
    generator_models: tuple[str, ...] = ()
    judge_models: tuple[str, ...] = ()
    tavily_api_key: SecretStr | None = None
    pexels_api_key: SecretStr | None = None
    openverse_api_key: SecretStr | None = None
    service_api_key: SecretStr | None = None
    cors_origins: tuple[str, ...] = ()
    log_level: str = DEFAULT_LOG_LEVEL

    @field_validator("discovery_model")
    @classmethod
    def _discovery_model_is_not_blank(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("discovery model must not be blank")
        return normalized

    @field_validator("llm_base_url")
    @classmethod
    def _llm_base_url_is_http(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError(f"{LLM_BASE_URL_ENV} must be an absolute HTTP(S) URL")
        return value.rstrip("/")

    @field_validator("cors_origins")
    @classmethod
    def _cors_origins_are_origins(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for value in values:
            parsed = urlsplit(value)
            if (
                parsed.scheme not in {"http", "https"}
                or not parsed.netloc
                or parsed.path not in {"", "/"}
                or parsed.query
                or parsed.fragment
            ):
                raise ValueError(
                    f"{CORS_ORIGINS_ENV} entries must be HTTP(S) origins without paths"
                )
        return tuple(value.rstrip("/") for value in values)

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        normalized = value.strip().upper() or DEFAULT_LOG_LEVEL
        if normalized not in VALID_LOG_LEVELS:
            expected = ", ".join(sorted(VALID_LOG_LEVELS))
            raise ValueError(f"{LOG_LEVEL_ENV} must be one of: {expected}")
        return normalized

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> "Settings":
        """Build a fresh snapshot without loading files or mutating the process."""
        source = os.environ if environ is None else environ
        return cls(
            discovery_model=_optional(source.get(DISCOVERY_MODEL_ENV)) or DEFAULT_DISCOVERY_MODEL,
            llm_base_url=_optional(source.get(LLM_BASE_URL_ENV)),
            llm_api_key=_optional(source.get(LLM_API_KEY_ENV)),
            generator_models=_csv(source.get(GENERATOR_MODELS_ENV)),
            judge_models=_csv(source.get(JUDGE_MODELS_ENV)),
            tavily_api_key=_optional(source.get(TAVILY_API_KEY_ENV)),
            pexels_api_key=_optional(source.get(PEXELS_API_KEY_ENV)),
            openverse_api_key=_optional(source.get(OPENVERSE_API_KEY_ENV)),
            service_api_key=_optional(source.get(SERVICE_API_KEY_ENV)),
            cors_origins=_csv(source.get(CORS_ORIGINS_ENV)),
            log_level=_optional(source.get(LOG_LEVEL_ENV)) or DEFAULT_LOG_LEVEL,
        )

    @staticmethod
    def reveal(secret: SecretStr | None) -> str | None:
        """Unwrap a secret only at the adapter boundary that needs it."""
        return secret.get_secret_value() if secret is not None else None

    @property
    def llm_api_key_value(self) -> str | None:
        return self.reveal(self.llm_api_key)

    @property
    def tavily_api_key_value(self) -> str | None:
        return self.reveal(self.tavily_api_key)

    @property
    def pexels_api_key_value(self) -> str | None:
        return self.reveal(self.pexels_api_key)

    @property
    def openverse_api_key_value(self) -> str | None:
        return self.reveal(self.openverse_api_key)

    @property
    def service_api_key_value(self) -> str | None:
        return self.reveal(self.service_api_key)


def get_settings(environ: Mapping[str, str] | None = None) -> Settings:
    """Return a fresh settings snapshot so tests and long-lived hosts stay predictable."""
    return Settings.from_env(environ)


def load_dotenv_file(path: str | Path | None = None, *, override: bool = False) -> bool:
    """Explicitly load a dotenv file for an application entry point."""
    from dotenv import load_dotenv

    return bool(load_dotenv(dotenv_path=path, override=override))
