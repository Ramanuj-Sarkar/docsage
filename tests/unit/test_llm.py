"""Tests for the chat-model factory."""

from __future__ import annotations

import pytest
from langchain_openai import ChatOpenAI

from docsage.config import Settings
from docsage.llm import FAKE_GRADE_MARKER, FAKE_GRADE_YES, ScriptedChatModel, get_chat_model


def test_fake_provider_returns_scripted_model() -> None:
    settings = Settings(llm_provider="fake")
    model = get_chat_model(settings)
    assert isinstance(model, ScriptedChatModel)


def test_fake_model_routes_grade_marker_to_valid_json() -> None:
    model = get_chat_model(Settings(llm_provider="fake"))
    response = model.invoke(f"You are a {FAKE_GRADE_MARKER} the documents.")
    assert response.content == FAKE_GRADE_YES


def test_openai_provider_returns_chat_openai() -> None:
    settings = Settings(llm_provider="openai", openai_api_key="sk-test")
    model = get_chat_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-4o-mini"


def test_openai_provider_without_explicit_key_uses_env(monkeypatch) -> None:
    # No key in settings -> falls back to the OPENAI_API_KEY environment var.
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    model = get_chat_model(Settings(llm_provider="openai"))
    assert isinstance(model, ChatOpenAI)


def test_unsupported_provider_raises_helpful_error() -> None:
    settings = Settings(llm_provider="anthropic")
    with pytest.raises(ValueError, match="anthropic"):
        get_chat_model(settings)
