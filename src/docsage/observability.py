"""Observability wiring for LangSmith (tracing + evals) and LangFuse (traces).

This module is the single entry point for constructing observability clients
and callbacks so tests can stub it. Both backends can be active at once:
LangSmith auto-traces via env vars; LangFuse gets its callback handler
attached to each graph invoke (``invoke_callbacks``).
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from langsmith import Client

from docsage.config import Settings, get_settings

if TYPE_CHECKING:
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler as LangfuseCallbackHandler

LANGSMITH_ENV_VARS: tuple[str, ...] = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",
)

LANGFUSE_ENV_VARS: tuple[str, ...] = (
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
)


def apply_tracing_env(settings: Settings | None = None) -> bool:
    """Apply LangSmith tracing env vars from settings.

    Returns True when LangSmith tracing is enabled. Idempotent; call once at
    process start (e.g. the CLI entry point) before any LangChain run.
    """
    settings = settings or get_settings()
    if not settings.langchain_tracing_v2:
        return False
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
    os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
    if settings.langchain_api_key:
        os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key
    return True


def get_langsmith_client(settings: Settings | None = None) -> Client:
    """Return a langsmith.Client configured from settings (lazy, no network)."""
    settings = settings or get_settings()
    return Client(
        api_url=settings.langchain_endpoint,
        api_key=settings.langchain_api_key,
    )


def apply_langfuse_env(settings: Settings | None = None) -> bool:
    """Apply LangFuse env vars from settings.

    Returns True when both keys are configured. The LangFuse LangChain
    callback handler reads the secret key and host from the environment, so
    this must be applied before constructing a handler.
    """
    settings = settings or get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return False
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host
    return True


def get_langfuse_handler(
    settings: Settings | None = None,
) -> LangfuseCallbackHandler | None:
    """Return a LangFuse LangChain callback handler, or None if not configured.

    The handler's ``get_client`` only *looks up* registered clients, so the
    client instance must be constructed first (which registers it). We do that
    here before constructing the handler.
    """
    settings = settings or get_settings()
    if not apply_langfuse_env(settings):
        return None
    # Deferred imports: keep the module importable without the LangFuse SDK.
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )
    return CallbackHandler(public_key=settings.langfuse_public_key)


def get_langfuse_client(settings: Settings | None = None) -> Langfuse | None:
    """Return a LangFuse client, or None if not configured (lazy, no network)."""
    settings = settings or get_settings()
    if not (settings.langfuse_public_key and settings.langfuse_secret_key):
        return None
    from langfuse import Langfuse  # deferred import

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def invoke_callbacks(
    settings: Settings | None = None,
) -> list[LangfuseCallbackHandler]:
    """Callbacks to attach to a graph invoke.

    Returns ``[LangFuse handler]`` when LangFuse is configured, else ``[]``.
    LangSmith needs no callback: it auto-traces via ``apply_tracing_env``.
    """
    handler = get_langfuse_handler(settings)
    return [handler] if handler is not None else []


def flush_langfuse(handler: LangfuseCallbackHandler | None) -> None:
    """Flush pending LangFuse events for the given callback handler.

    The handler's client batches trace ingestion asynchronously; call this
    after a run (tests and CLI) so traces are sent promptly instead of on the
    client's internal flush interval.
    """
    if handler is None:
        return
    handler._langfuse_client.flush()  # noqa: SLF001 - wrapper around the SDK's internal client
