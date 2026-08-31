"""Retrieval: embeddings + in-memory vector store + retriever builder.

Uses the pure-Python :class:`InMemoryVectorStore` so the project runs on
Python 3.14 without native vector-store wheels (Chroma/FAISS are not yet
3.14-ready). For production, swap ``build_retriever`` internals for pgvector
or Qdrant without changing its signature.
"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.vectorstores.in_memory import InMemoryVectorStore

from docsage.config import Settings, get_settings
from docsage.embeddings import HashEmbeddings


def get_embeddings(settings: Settings | None = None) -> Embeddings:
    """Return an embeddings model for the configured provider.

    ``fake`` (default for tests/offline) yields deterministic hash-based
    vectors; ``openai`` uses OpenAI embeddings.
    """
    settings = settings or get_settings()
    if settings.llm_provider == "fake":
        return HashEmbeddings(size=1536)

    # Deferred import: keeps the module importable without the OpenAI SDK.
    from langchain_openai import OpenAIEmbeddings

    kwargs: dict[str, object] = {"model": "text-embedding-3-small"}
    if settings.openai_api_key:
        kwargs["api_key"] = settings.openai_api_key
    return OpenAIEmbeddings(**kwargs)


def build_retriever(
    documents: Sequence[Document],
    *,
    embeddings: Embeddings | None = None,
    k: int = 4,
) -> VectorStoreRetriever:
    """Build a retriever over ``documents`` returning the top-``k`` matches."""
    store = InMemoryVectorStore(embedding=embeddings or get_embeddings())
    store.add_documents(list(documents))
    return store.as_retriever(search_kwargs={"k": k})
