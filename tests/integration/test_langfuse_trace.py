"""Integration: dual instrumentation — one run traces to both LangSmith and LangFuse.

Requires OPENAI_API_KEY, LANGCHAIN_API_KEY, LANGFUSE_PUBLIC_KEY, and
LANGFUSE_SECRET_KEY (env or .env). Skipped otherwise.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from langchain_core.documents import Document

from docsage.config import get_settings
from docsage.embeddings import HashEmbeddings
from docsage.graph.build import build_graph
from docsage.llm import get_chat_model
from docsage.observability import (
    apply_langfuse_env,
    apply_tracing_env,
    flush_langfuse,
    get_langfuse_client,
    get_langfuse_handler,
    get_langsmith_client,
)
from docsage.retrieval import build_retriever

pytestmark = pytest.mark.integration

DOCS = [
    Document(
        page_content="LangGraph is a library for building stateful agents.",
        metadata={"title": "langgraph-doc"},
    )
]

_CHECKED_CREDENTIALS = {
    "OPENAI_API_KEY": "openai_api_key",
    "LANGCHAIN_API_KEY": "langchain_api_key",
    "LANGFUSE_PUBLIC_KEY": "langfuse_public_key",
    "LANGFUSE_SECRET_KEY": "langfuse_secret_key",
}


def _skip_unless_credentials() -> None:
    settings = get_settings()
    missing = [
        name
        for name, field in _CHECKED_CREDENTIALS.items()
        if not (os.environ.get(name) or getattr(settings, field))
    ]
    if missing:
        pytest.skip(f"Missing credentials: {', '.join(missing)}")


def test_single_run_traces_to_both_platforms() -> None:
    _skip_unless_credentials()
    assert apply_tracing_env(), "LangSmith tracing is not enabled in settings"
    assert apply_langfuse_env(), "LangFuse credentials are incomplete"

    retriever = build_retriever(DOCS, embeddings=HashEmbeddings(size=256), k=1)
    graph = build_graph(get_chat_model(), retriever)
    handler = get_langfuse_handler()
    run_id = uuid.uuid4()

    result = graph.invoke(
        {"question": "What is LangGraph?"},
        config={
            "configurable": {"thread_id": f"dual-{run_id.hex[:8]}"},
            "callbacks": [handler] if handler else [],
            "run_id": run_id,
        },
    )
    assert result.get("final_answer"), "expected a final answer from the real model"
    assert handler is not None and handler.last_trace_id, (
        "LangFuse handler did not capture a trace id"
    )
    trace_id = handler.last_trace_id
    # The handler batches ingestion asynchronously; push it out immediately.
    flush_langfuse(handler)

    # LangFuse: traces are ingested asynchronously; poll for up to a minute.
    langfuse_client = get_langfuse_client()
    listed = []
    for _ in range(15):
        listed = langfuse_client.api.trace.list(limit=20).data
        if any(getattr(trace, "id", None) == trace_id for trace in listed):
            break
        time.sleep(4)
    assert any(getattr(trace, "id", None) == trace_id for trace in listed), (
        f"trace {trace_id} not found in LangFuse"
    )

    # LangSmith: the same run (same run_id) also produced a LangSmith trace.
    langsmith_client = get_langsmith_client()
    runs = []
    for _ in range(10):
        runs = list(langsmith_client.list_runs(run_ids=[run_id]))
        if runs:
            break
        time.sleep(3)
    assert runs, f"run {run_id} not found in LangSmith project {get_settings().langchain_project!r}"
    assert runs[0].name == "LangGraph"
