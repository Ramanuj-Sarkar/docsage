"""Direct unit tests for the individual graph nodes (pure functions)."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from docsage.embeddings import HashEmbeddings
from docsage.graph.edges import ROUTE_GENERATE, ROUTE_REWRITE
from docsage.graph.nodes import finalize, generate, grade, retrieve, rewrite
from docsage.prompts import GradeDocuments
from docsage.retrieval import build_retriever

DOCS = [
    Document(
        page_content="LangGraph builds stateful agents.",
        metadata={"title": "lg"},
    )
]


def _state(**overrides: object) -> dict:
    base: dict[str, object] = {
        "question": "what is langgraph?",
        "search_query": "",
        "documents": [],
        "grade_decision": "",
        "generation": "",
        "attempts": 0,
        "final_answer": "",
    }
    base.update(overrides)
    return base


class _RecordingChain:
    """Stub Runnable that records its inputs and returns a scripted result."""

    def __init__(self, result: object) -> None:
        self.result = result
        self.last_input: dict | None = None

    def invoke(self, inputs: dict, config: object | None = None) -> object:
        self.last_input = inputs
        return self.result


def _retriever() -> object:
    return build_retriever(DOCS, embeddings=HashEmbeddings(size=64), k=4)


def _assert_same_docs(actual: object, expected: list[Document]) -> None:
    # The vector store assigns Document.id on add; compare content and metadata.
    assert [d.page_content for d in actual] == [d.page_content for d in expected]
    assert [d.metadata for d in actual] == [d.metadata for d in expected]


def test_retrieve_uses_search_query_when_set() -> None:
    out = retrieve(_state(search_query="rewritten"), _retriever())
    _assert_same_docs(out["documents"], DOCS)


def test_retrieve_falls_back_to_question() -> None:
    out = retrieve(_state(question="LangGraph agents?"), _retriever())
    _assert_same_docs(out["documents"], DOCS)


def test_grade_keeps_relevant_docs_and_routes_to_generate() -> None:
    chain = _RecordingChain(GradeDocuments(binary_score="yes"))
    out = grade(_state(documents=DOCS), chain)
    assert out["grade_decision"] == ROUTE_GENERATE
    assert out["documents"] == DOCS


def test_grade_with_all_irrelevant_routes_to_rewrite() -> None:
    chain = _RecordingChain(GradeDocuments(binary_score="no"))
    out = grade(_state(documents=DOCS), chain)
    assert out["grade_decision"] == ROUTE_REWRITE
    assert out["documents"] == []


def test_grade_with_no_documents_routes_to_rewrite() -> None:
    chain = _RecordingChain(GradeDocuments(binary_score="yes"))
    out = grade(_state(), chain)
    assert out["grade_decision"] == ROUTE_REWRITE


class _BrokenChain(_RecordingChain):
    """Stub chain that raises like the real parser on invalid LLM output."""

    def __init__(self) -> None:
        super().__init__(None)

    def invoke(self, inputs: dict, config: object | None = None) -> object:
        self.last_input = inputs
        from langchain_core.exceptions import OutputParserException

        raise OutputParserException("Invalid json output: Yes")


def test_grade_handles_unparseable_output_as_irrelevant() -> None:
    out = grade(_state(documents=DOCS), _BrokenChain())
    assert out["grade_decision"] == ROUTE_REWRITE
    assert out["documents"] == []


def test_rewrite_updates_query_and_increments_attempts() -> None:
    chain = _RecordingChain(AIMessage(content="better query"))
    out = rewrite(_state(attempts=1), chain)
    assert out["search_query"] == "better query"
    assert out["attempts"] == 2


def test_generate_passes_formatted_context() -> None:
    chain = _RecordingChain(AIMessage(content="answer text"))
    out = generate(_state(documents=DOCS, question="q?"), chain)
    assert out["generation"] == "answer text"
    assert "LangGraph builds stateful agents." in chain.last_input["context"]
    assert chain.last_input["question"] == "q?"


def test_finalize_promotes_generation() -> None:
    out = finalize(_state(generation="answer"))
    assert out["final_answer"] == "answer"
