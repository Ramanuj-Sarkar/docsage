"""Document loading for the corpus used by the DocSage agent."""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".md", ".txt"})

DEFAULT_DOCS: list[Document] = [
    Document(
        page_content=(
            "LangGraph is a library for building stateful agents with graphs. "
            "It provides nodes, edges, conditional routing, and checkpoints."
        ),
        metadata={"title": "langgraph.md"},
    ),
    Document(
        page_content=(
            "LangSmith traces and evaluates LLM applications. It records runs, "
            "supports dataset-based evaluation, and tracks experiments."
        ),
        metadata={"title": "langsmith.md"},
    ),
    Document(
        page_content=(
            "LangFuse provides LLM observability with traces, spans, and "
            "scores for monitoring and cost tracking."
        ),
        metadata={"title": "langfuse.md"},
    ),
]


def load_document_files(directory: str | Path) -> list[Document]:
    """Load every ``.md``/``.txt`` file under ``directory`` (recursive).

    Each document's ``title`` metadata is its path relative to ``directory``.
    Empty files are skipped. A missing directory yields an empty list.
    """
    root = Path(directory)
    if not root.is_dir():
        return []
    documents: list[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={"title": str(path.relative_to(root))},
                    )
                )
    return documents


def load_app_documents(directory: str | Path | None = None) -> list[Document]:
    """Return corpus documents, falling back to the built-in fixture docs."""
    if directory:
        documents = load_document_files(directory)
        if documents:
            return documents
    return DEFAULT_DOCS
