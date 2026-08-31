"""Smoke tests for the Phase 0 scaffold: settings load and pytest wiring works."""

from __future__ import annotations

from docsage.config import Settings, get_settings


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.llm_provider == "openai"
    assert settings.langchain_project == "docsage"
    assert settings.max_retries == 2
    # No observability credentials by default -> tracing disabled.
    assert settings.tracing_enabled is False


def test_settings_from_environment(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
    settings = Settings()
    assert settings.llm_provider == "fake"
    assert settings.tracing_enabled is True


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
