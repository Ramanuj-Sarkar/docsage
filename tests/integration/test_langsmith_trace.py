"""Integration: a real graph run produces a trace in the LangSmith project.

Requires OPENAI_API_KEY and LANGCHAIN_API_KEY (env or .env). Skipped otherwise.
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
from docsage.observability import apply_tracing_env, get_langsmith_client
from docsage.retrieval import build_retriever

pytestmark = pytest.mark.integration

DOCS = [
    Document(
        page_content="LangGraph is a library for building stateful agents.",
        metadata={"title": "langgraph-doc"},
    )
]


def _skip_unless_credentials() -> None:
    settings = get_settings()
    missing = []
    if not (os.environ.get("OPENAI_API_KEY") or settings.openai_api_key):
        missing.append("OPENAI_API_KEY")
    if not (os.environ.get("LANGCHAIN_API_KEY") or settings.langchain_api_key):
        missing.append("LANGCHAIN_API_KEY")
    if missing:
        pytest.skip(f"Missing credentials: {', '.join(missing)}")


def test_real_run_creates_langsmith_trace() -> None:
    _skip_unless_credentials()
    assert apply_tracing_env(), "LangSmith tracing is not enabled in settings"

    retriever = build_retriever(DOCS, embeddings=HashEmbeddings(size=256), k=1)
    graph = build_graph(get_chat_model(), retriever)
    run_id = uuid.uuid4()
    result = graph.invoke(
        {"question": "What is LangGraph?"},
        config={
            "configurable": {"thread_id": f"integration-{run_id.hex[:8]}"},
            "run_id": run_id,
        },
    )
    assert result.get("final_answer"), "expected a final answer from the real model"

    # LangSmith batches trace uploads asynchronously; poll for the exact run id.
    client = get_langsmith_client()
    runs = []
    for _ in range(10):
        runs = list(client.list_runs(run_ids=[run_id]))
        if runs:
            break
        time.sleep(3)
    assert runs, f"run {run_id} not found in LangSmith project {get_settings().langchain_project!r}"
    assert runs[0].name == "LangGraph"
