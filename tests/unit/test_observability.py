"""Tests for observability wiring (LangSmith)."""

from __future__ import annotations

import os

from docsage.config import Settings
from docsage.observability import (
    LANGSMITH_ENV_VARS,
    apply_tracing_env,
    get_langsmith_client,
)


def test_apply_tracing_env_disabled_returns_false() -> None:
    assert apply_tracing_env(Settings(langchain_tracing_v2=False)) is False


def test_apply_tracing_env_enabled_sets_env_vars() -> None:
    settings = Settings(
        langchain_tracing_v2=True,
        langchain_api_key="sk-test",
        langchain_project="docsage-test",
        langchain_endpoint="https://api.smith.langchain.com",
    )
    saved = {name: os.environ.get(name) for name in LANGSMITH_ENV_VARS}
    try:
        assert apply_tracing_env(settings) is True
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
        assert os.environ["LANGCHAIN_PROJECT"] == "docsage-test"
        assert os.environ["LANGCHAIN_ENDPOINT"] == "https://api.smith.langchain.com"
        assert os.environ["LANGCHAIN_API_KEY"] == "sk-test"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_apply_tracing_env_without_api_key_skips_key_var() -> None:
    settings = Settings(
        langchain_tracing_v2=True,
        langchain_api_key=None,
        langchain_project="docsage-test",
    )
    saved = {name: os.environ.get(name) for name in LANGSMITH_ENV_VARS}
    try:
        assert apply_tracing_env(settings) is True
        assert "LANGCHAIN_API_KEY" not in os.environ
        assert os.environ["LANGCHAIN_TRACING_V2"] == "true"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_get_langsmith_client_uses_settings() -> None:
    client = get_langsmith_client(
        Settings(
            langchain_api_key="sk-test",
            langchain_endpoint="https://api.smith.langchain.com",
        )
    )
    assert client.api_key == "sk-test"
    assert client.api_url == "https://api.smith.langchain.com"
