"""Tests for eval dataset loading."""

from __future__ import annotations

import json

import pytest

from docsage.datasets import load_qa_dataset


def test_load_qa_dataset_parses_rows(tmp_path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(
        '{"question": "q1", "answer": "a1"}\n\n{"question": "q2", "answer": "a2"}\n',
        encoding="utf-8",
    )
    assert load_qa_dataset(path) == [
        {"question": "q1", "answer": "a1"},
        {"question": "q2", "answer": "a2"},
    ]


def test_load_qa_dataset_ignores_blank_lines(tmp_path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text('\n{"question": "q", "answer": "a"}\n\n', encoding="utf-8")
    assert len(load_qa_dataset(path)) == 1


def test_load_qa_dataset_rejects_missing_key(tmp_path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text(json.dumps({"question": "q only"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="answer"):
        load_qa_dataset(path)


def test_load_qa_dataset_rejects_invalid_json(tmp_path) -> None:
    path = tmp_path / "qa.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        load_qa_dataset(path)
