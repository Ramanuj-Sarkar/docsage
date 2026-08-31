"""Tests for prompt templates and structured-output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from docsage.prompts import GradeDocuments, grade_prompt, rag_prompt, rewrite_prompt


def test_rag_prompt_renders_context_and_question() -> None:
    rendered = rag_prompt.format(
        context="Doc A: LangGraph is a graph library.",
        question="What is LangGraph?",
    )
    assert "Doc A: LangGraph is a graph library." in rendered
    assert "What is LangGraph?" in rendered


def test_grade_prompt_renders_document_and_question() -> None:
    rendered = grade_prompt.format(document="chunk text", question="q?")
    assert "chunk text" in rendered
    assert "q?" in rendered


def test_rewrite_prompt_renders_question() -> None:
    rendered = rewrite_prompt.format(question="tell me about checkpoints")
    assert "tell me about checkpoints" in rendered


def test_grade_documents_accepts_valid_scores() -> None:
    assert GradeDocuments(binary_score="yes").binary_score == "yes"
    assert GradeDocuments(binary_score="no").binary_score == "no"


def test_grade_documents_rejects_invalid_score() -> None:
    with pytest.raises(ValidationError):
        GradeDocuments(binary_score="maybe")
