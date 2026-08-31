"""Eval suite: LangSmith dataset evaluation of the DocSage agent.

Requires OPENAI_API_KEY and LANGCHAIN_API_KEY (env or .env). Skipped otherwise.
"""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from pathlib import Path

import pytest
from langsmith.evaluation import evaluate

from docsage.config import get_settings
from docsage.datasets import ensure_langsmith_dataset
from docsage.documents import DEFAULT_DOCS
from docsage.evals import (
    contains_reference_evaluator,
    exact_match_evaluator,
    llm_judge_evaluator,
)
from docsage.graph.build import build_graph
from docsage.llm import get_chat_model
from docsage.observability import apply_tracing_env, get_langsmith_client
from docsage.prompts import judge_prompt
from docsage.retrieval import build_retriever

pytestmark = pytest.mark.eval

DATASET_NAME = "docsage-qa"
DATASET_PATH = Path(__file__).resolve().parents[2] / "datasets" / "qa_pairs.jsonl"

FIXTURE_DOCS = DEFAULT_DOCS


def _skip_unless_credentials() -> None:
    settings = get_settings()
    missing = []
    if not (os.environ.get("OPENAI_API_KEY") or settings.openai_api_key):
        missing.append("OPENAI_API_KEY")
    if not (os.environ.get("LANGCHAIN_API_KEY") or settings.langchain_api_key):
        missing.append("LANGCHAIN_API_KEY")
    if missing:
        pytest.skip(f"Missing credentials: {', '.join(missing)}")


def _make_target() -> Callable[[dict], dict]:
    """Build the agent under evaluation (real model, real retrieval)."""
    retriever = build_retriever(FIXTURE_DOCS, k=3)
    graph = build_graph(get_chat_model(), retriever)

    def _target(example: dict) -> dict:
        # The compiled graph uses a checkpointer and requires a thread_id.
        result = graph.invoke(
            {"question": example["question"]},
            config={"configurable": {"thread_id": f"langsmith-eval-{uuid.uuid4().hex[:8]}"}},
        )
        return {"answer": result.get("final_answer", "")}

    return _target


def test_langsmith_dataset_eval() -> None:
    _skip_unless_credentials()
    assert apply_tracing_env(), "LangSmith tracing is not enabled in settings"

    client = get_langsmith_client()
    dataset_name = ensure_langsmith_dataset(client, DATASET_NAME, DATASET_PATH)

    results = evaluate(
        _make_target(),
        data=dataset_name,
        evaluators=[
            exact_match_evaluator,
            contains_reference_evaluator,
            llm_judge_evaluator(judge_prompt | get_chat_model()),
        ],
        experiment_prefix="docsage",
        client=client,
    )
    rows = list(results)
    assert rows, "expected at least one evaluated example"

    scores: list[tuple[str, float]] = []
    for row in rows:
        # evaluation_results is {"results": [EvaluationResult, ...]}.
        results = (row.get("evaluation_results") or {}).get("results", [])
        for result in results:
            scores.append((result.key, float(result.score)))

    keys = {key for key, _ in scores}
    assert {"exact_match", "contains_reference", "llm_judge"} <= keys, (
        f"expected all three evaluators to produce scores, got {keys}"
    )
    assert all(score in (0.0, 1.0) for _, score in scores)
