"""Behavioral tests for the compiled DocSage LangGraph agent."""

from __future__ import annotations

import json

import pytest
from langchain_core.documents import Document

from docsage.config import Settings
from docsage.embeddings import HashEmbeddings
from docsage.graph.build import _build_state_graph, build_graph
from docsage.llm import ScriptedChatModel
from docsage.retrieval import build_retriever

DOCS = [
    Document(
        page_content="LangGraph is a library for building stateful agents with graphs.",
        metadata={"title": "langgraph-doc"},
    ),
    Document(
        page_content="LangSmith traces and evaluates LLM applications.",
        metadata={"title": "langsmith-doc"},
    ),
]
SINGLE_DOC = [DOCS[0]]

GRADE_YES = json.dumps({"binary_score": "yes", "explanation": "relevant"})
GRADE_NO = json.dumps({"binary_score": "no", "explanation": "irrelevant"})


@pytest.fixture
def retriever() -> object:
    return build_retriever(DOCS, embeddings=HashEmbeddings(size=64), k=2)


@pytest.fixture
def single_doc_retriever() -> object:
    return build_retriever(SINGLE_DOC, embeddings=HashEmbeddings(size=64), k=1)


def _cfg(thread_id: str = "test-thread") -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _graph(
    retriever: object,
    grade_seq: tuple[str, ...],
    generate_seq: tuple[str, ...],
    rewrite_seq: tuple[str, ...] = ("rewritten query",),
    max_retries: int = 2,
) -> object:
    model = ScriptedChatModel(
        responses={
            "grader assessing": list(grade_seq),
            "query rewriter": list(rewrite_seq),
            "DocSage": list(generate_seq),
        }
    )
    return build_graph(model, retriever, Settings(max_retries=max_retries))


def test_happy_path_generates_answer(retriever: object) -> None:
    graph = _graph(
        retriever,
        grade_seq=(GRADE_YES,),
        generate_seq=("LangGraph is a graph library.",),
    )
    result = graph.invoke({"question": "What is LangGraph?"}, config=_cfg())
    assert result["final_answer"] == "LangGraph is a graph library."
    # `attempts` is only written when a rewrite happens; absent on the happy path.
    assert result.get("attempts", 0) == 0
    assert result.get("grade_decision") == "generate"
    assert len(result.get("documents", [])) == 2


def test_rewrite_loop_until_relevant(single_doc_retriever: object) -> None:
    graph = _graph(
        single_doc_retriever,
        grade_seq=(GRADE_NO, GRADE_YES),
        generate_seq=("The answer.",),
        rewrite_seq=("better query",),
    )
    result = graph.invoke({"question": "q"}, config=_cfg())
    assert result["attempts"] == 1
    assert result["search_query"] == "better query"
    assert result["final_answer"] == "The answer."


def test_retry_cap_stops_infinite_rewrite_loop(
    single_doc_retriever: object,
) -> None:
    graph = _graph(
        single_doc_retriever,
        grade_seq=(GRADE_NO,),
        generate_seq=("No relevant documents found.",),
        max_retries=2,
    )
    result = graph.invoke({"question": "q"}, config=_cfg())
    assert result["attempts"] == 2
    assert result["final_answer"] == "No relevant documents found."


def test_checkpointer_isolates_threads(retriever: object) -> None:
    graph = _graph(
        retriever,
        grade_seq=(GRADE_YES,),
        generate_seq=("answer one",),
    )
    cfg1 = _cfg("thread-1")
    cfg2 = _cfg("thread-2")

    result1 = graph.invoke({"question": "q1"}, config=cfg1)
    assert graph.get_state(cfg1).values.get("final_answer") == result1["final_answer"]
    # A fresh thread has no state yet.
    assert graph.get_state(cfg2).values.get("final_answer") is None

    graph.invoke({"question": "q2"}, config=cfg2)
    assert graph.get_state(cfg2).values["question"] == "q2"


def test_graph_topology(retriever: object) -> None:
    model = ScriptedChatModel(responses={"grader assessing": [GRADE_YES]})
    graph = _build_state_graph(model, retriever, Settings())

    assert set(graph.nodes) == {"retrieve", "grade", "rewrite", "generate", "finalize"}
    static_edges = {
        ("__start__", "retrieve"),
        ("retrieve", "grade"),
        ("rewrite", "retrieve"),
        ("generate", "finalize"),
        ("finalize", "__end__"),
    }
    assert static_edges <= set(graph.edges)
    # The only conditional branch point is after grading.
    assert set(graph.branches) == {"grade"}
