"""Tests for LangFuse observability wiring (offline, no network)."""

from __future__ import annotations

import os

from docsage.config import Settings
from docsage.observability import (
    LANGFUSE_ENV_VARS,
    apply_langfuse_env,
    flush_langfuse,
    get_langfuse_client,
    get_langfuse_handler,
    invoke_callbacks,
)


def _langfuse_settings() -> Settings:
    return Settings(
        langfuse_public_key="pk-test",
        langfuse_secret_key="sk-test",
        langfuse_host="https://cloud.langfuse.com",
    )


def test_apply_langfuse_env_disabled_returns_false() -> None:
    assert apply_langfuse_env(Settings()) is False


def test_apply_langfuse_env_requires_both_keys() -> None:
    assert apply_langfuse_env(Settings(langfuse_public_key="pk-only")) is False


def test_apply_langfuse_env_sets_env_vars() -> None:
    saved = {name: os.environ.get(name) for name in LANGFUSE_ENV_VARS}
    try:
        assert apply_langfuse_env(_langfuse_settings()) is True
        assert os.environ["LANGFUSE_PUBLIC_KEY"] == "pk-test"
        assert os.environ["LANGFUSE_SECRET_KEY"] == "sk-test"
        assert os.environ["LANGFUSE_HOST"] == "https://cloud.langfuse.com"
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def test_get_langfuse_handler_none_without_keys() -> None:
    assert get_langfuse_handler(Settings()) is None


def test_get_langfuse_handler_returns_handler_with_keys() -> None:
    handler = get_langfuse_handler(_langfuse_settings())
    assert handler is not None
    from langfuse.langchain import CallbackHandler

    assert isinstance(handler, CallbackHandler)


def test_get_langfuse_client_none_without_keys() -> None:
    assert get_langfuse_client(Settings()) is None


def test_get_langfuse_client_returns_client_with_keys() -> None:
    client = get_langfuse_client(_langfuse_settings())
    assert client is not None
    from langfuse import Langfuse

    assert isinstance(client, Langfuse)


def test_invoke_callbacks_empty_without_keys() -> None:
    assert invoke_callbacks(Settings()) == []


def test_invoke_callbacks_contains_handler_with_keys() -> None:
    callbacks = invoke_callbacks(_langfuse_settings())
    assert len(callbacks) == 1


def test_flush_langfuse_is_noop_for_none() -> None:
    flush_langfuse(None)  # must not raise


def test_flush_langfuse_flushes_handler_client() -> None:
    handler = get_langfuse_handler(_langfuse_settings())
    assert handler is not None
    flush_langfuse(handler)  # offline-safe: dummy credentials, no network
