"""Tests for the deterministic hash-based fake embeddings."""

from __future__ import annotations

from docsage.embeddings import HashEmbeddings


def test_embeddings_are_deterministic() -> None:
    emb = HashEmbeddings(size=64)
    assert emb.embed_query("LangGraph agents") == emb.embed_query("LangGraph agents")


def test_embeddings_differentiate_unrelated_texts() -> None:
    emb = HashEmbeddings(size=64)
    assert emb.embed_query("LangGraph agents") != emb.embed_query("recipe for pasta")


def test_embeddings_are_unit_normalized() -> None:
    emb = HashEmbeddings(size=64)
    vector = emb.embed_query("some text")
    norm = sum(v * v for v in vector) ** 0.5
    assert abs(norm - 1.0) < 1e-9


def test_embed_documents_returns_parallel_list() -> None:
    emb = HashEmbeddings(size=64)
    vectors = emb.embed_documents(["a", "b", "c"])
    assert len(vectors) == 3
    assert all(len(v) == 64 for v in vectors)


def test_similar_texts_are_more_similar_than_unrelated() -> None:
    emb = HashEmbeddings(size=256)
    v1 = emb.embed_query("langgraph stateful agents")
    v2 = emb.embed_query("langgraph agent state")
    v3 = emb.embed_query("banana bread recipe")
    sim_ab = sum(a * b for a, b in zip(v1, v2, strict=True))
    sim_ac = sum(a * c for a, c in zip(v1, v3, strict=True))
    assert sim_ab > sim_ac
