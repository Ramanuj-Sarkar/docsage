"""Shared eval dataset loading and LangSmith dataset seeding."""

from __future__ import annotations

import json
from pathlib import Path

from langsmith import Client


def load_qa_dataset(path: str | Path) -> list[dict]:
    """Load a JSONL dataset of ``{"question": ..., "answer": ...}`` rows.

    Blank lines are ignored; malformed rows raise :class:`ValueError`.
    """
    rows: list[dict] = []
    for line_number, line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(), start=1
    ):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        if not isinstance(row, dict) or not {"question", "answer"} <= row.keys():
            raise ValueError(
                f"Line {line_number} of {path} must be an object with "
                f"'question' and 'answer': {row!r}"
            )
        rows.append(row)
    return rows


def ensure_langsmith_dataset(client: Client, dataset_name: str, path: str | Path) -> str:
    """Create the LangSmith dataset from a JSONL file if missing, then backfill new rows.

    Idempotent: rows whose question already exists in the dataset are skipped.
    Returns the dataset name.
    """
    try:
        dataset = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        dataset = client.create_dataset(dataset_name=dataset_name)

    existing = {
        example.inputs.get("question") for example in client.list_examples(dataset_id=dataset.id)
    }
    new_rows = [row for row in load_qa_dataset(path) if row["question"] not in existing]
    if new_rows:
        client.create_examples(
            dataset_id=dataset.id,
            examples=[
                {
                    "inputs": {"question": row["question"]},
                    "outputs": {"answer": row["answer"]},
                }
                for row in new_rows
            ],
        )
    return dataset_name
