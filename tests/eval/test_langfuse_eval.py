"""Eval suite: LangFuse traces + scoring over the shared QA dataset.

Runs the agent over ``datasets/qa_pairs.jsonl`` with the LangFuse handler
attached, scores each trace, and asserts the traces were ingested.

Requires OPENAI_API_KEY, LANGFUSE_PUBLIC_KEY, and LANGFUSE_SECRET_KEY
(env or .env). Skipped otherwise.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from docsage.config import get_settings
from docsage.datasets import load_qa_dataset
from docsage.documents import DEFAULT_DOCS
from docsage.embeddings import HashEmbeddings
from docsage.graph.build import build_graph
from docsage.llm import get_chat_model
from docsage.observability import (
    apply_langfuse_env,
    flush_langfuse,
    get_langfuse_client,
    get_langfuse_handler,
)
from docsage.retrieval import build_retriever

pytestmark = pytest.mark.eval

DATASET_PATH = Path(__file__).resolve().parents[2] / "datasets" / "qa_pairs.jsonl"

FIXTURE_DOCS = DEFAULT_DOCS

_CHECKED_CREDENTIALS = {
    "OPENAI_API_KEY": "openai_api_key",
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


def test_langfuse_traces_and_scores() -> None:
    _skip_unless_credentials()
    assert apply_langfuse_env(), "LangFuse credentials are incomplete"

    retriever = build_retriever(FIXTURE_DOCS, embeddings=HashEmbeddings(size=256), k=3)
    graph = build_graph(get_chat_model(), retriever)
    handler = get_langfuse_handler()
    client = get_langfuse_client()
    assert handler is not None and client is not None

    rows = load_qa_dataset(DATASET_PATH)
    assert rows, "expected a non-empty dataset"

    trace_ids: list[str] = []
    for index, row in enumerate(rows):
        result = graph.invoke(
            {"question": row["question"]},
            config={
                "configurable": {"thread_id": f"langfuse-eval-{index}"},
                "callbacks": [handler],
            },
        )
        assert handler.last_trace_id, f"no trace captured for row {index}"
        trace_id = handler.last_trace_id
        trace_ids.append(trace_id)

        prediction = result.get("final_answer", "")
        client.create_score(
            trace_id=trace_id,
            name="exact_match",
            value=float(prediction.strip() == row["answer"].strip()),
        )

    client.flush()
    # The handler batches trace ingestion separately; flush it too.
    flush_langfuse(handler)

    # All traces landed (poll for up to a minute for async ingestion).
    listed = []
    for _ in range(15):
        listed = client.api.trace.list(limit=50).data
        if len([t for t in listed if getattr(t, "id", None) in trace_ids]) == len(trace_ids):
            break
        time.sleep(4)
    ingested = {getattr(trace, "id", None) for trace in listed}
    assert set(trace_ids) <= ingested, "not all eval traces were ingested into LangFuse"
