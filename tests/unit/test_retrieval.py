"""Tests for retrieval: embeddings and the in-memory retriever."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from docsage.config import Settings
from docsage.embeddings import HashEmbeddings
from docsage.retrieval import build_retriever, get_embeddings

DOCS = [
    Document(
        page_content="LangGraph is a library for building stateful agents.",
        metadata={"title": "langgraph-doc"},
    ),
    Document(
        page_content="LangSmith traces LLM applications.",
        metadata={"title": "langsmith-doc"},
    ),
    Document(
        page_content="LangFuse provides LLM observability.",
        metadata={"title": "langfuse-doc"},
    ),
]
EMBEDDINGS = HashEmbeddings(size=1536)


def test_get_embeddings_fake_for_fake_provider() -> None:
    assert isinstance(get_embeddings(Settings(llm_provider="fake")), HashEmbeddings)


def test_get_embeddings_openai_for_openai_provider() -> None:
    # Construction only — offline-safe; network happens at first call.
    embeddings = get_embeddings(Settings(llm_provider="openai", openai_api_key="sk-test"))
    assert isinstance(embeddings, OpenAIEmbeddings)


def test_get_embeddings_openai_falls_back_to_env_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-env")
    embeddings = get_embeddings(Settings(llm_provider="openai"))
    assert isinstance(embeddings, OpenAIEmbeddings)


def test_build_retriever_returns_documents_with_metadata() -> None:
    retriever = build_retriever(DOCS, embeddings=EMBEDDINGS, k=2)
    results = retriever.invoke("LangGraph stateful agents")
    assert 1 <= len(results) <= 2
    assert all(isinstance(d, Document) for d in results)
    assert all("title" in d.metadata for d in results)


def test_build_retriever_on_empty_store() -> None:
    retriever = build_retriever([], embeddings=EMBEDDINGS, k=2)
    assert retriever.invoke("anything") == []


def test_build_retriever_respects_k() -> None:
    retriever = build_retriever(DOCS, embeddings=EMBEDDINGS, k=1)
    assert len(retriever.invoke("LangGraph")) == 1
