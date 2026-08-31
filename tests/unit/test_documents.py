"""Tests for corpus document loading."""

from __future__ import annotations

from langchain_core.documents import Document

from docsage.documents import DEFAULT_DOCS, load_app_documents, load_document_files


def test_load_document_files_reads_md_and_txt(tmp_path) -> None:
    (tmp_path / "a.md").write_text("Alpha content\n", encoding="utf-8")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b.txt").write_text("Beta content", encoding="utf-8")
    (tmp_path / "skip.py").write_text("not a doc", encoding="utf-8")

    docs = load_document_files(tmp_path)
    assert [d.page_content for d in docs] == ["Alpha content", "Beta content"]
    assert [d.metadata["title"] for d in docs] == ["a.md", "sub/b.txt"]


def test_load_document_files_skips_empty_files(tmp_path) -> None:
    (tmp_path / "empty.md").write_text("   \n", encoding="utf-8")
    (tmp_path / "full.md").write_text("full", encoding="utf-8")

    docs = load_document_files(tmp_path)
    assert len(docs) == 1
    assert docs[0].page_content == "full"


def test_load_document_files_missing_directory() -> None:
    assert load_document_files("/nonexistent/path") == []


def test_load_app_documents_falls_back_to_defaults(tmp_path) -> None:
    docs = load_app_documents(tmp_path / "missing")
    assert docs == DEFAULT_DOCS
    assert all(isinstance(d, Document) for d in docs)


def test_load_app_documents_prefers_corpus(tmp_path) -> None:
    (tmp_path / "custom.md").write_text("custom doc", encoding="utf-8")
    docs = load_app_documents(tmp_path)
    assert len(docs) == 1
    assert docs[0].page_content == "custom doc"
