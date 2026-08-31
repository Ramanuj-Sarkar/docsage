"""DocSage CLI: ``ask`` the agent questions and ``seed`` the corpus.

The CLI is a thin wrapper over the graph builder and observability wiring so
the exact same code paths are exercised by the tests.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from langgraph.graph.state import CompiledStateGraph

from docsage.config import Settings, get_settings
from docsage.documents import DEFAULT_DOCS, load_app_documents
from docsage.graph.build import build_graph
from docsage.llm import get_chat_model
from docsage.observability import (
    apply_langfuse_env,
    apply_tracing_env,
    get_langfuse_client,
    get_langfuse_handler,
    invoke_callbacks,
)
from docsage.retrieval import build_retriever, get_embeddings

_SAMPLE_CORPUS: dict[str, str] = {
    "langgraph.md": DEFAULT_DOCS[0].page_content,
    "langsmith.md": DEFAULT_DOCS[1].page_content,
    "langfuse.md": DEFAULT_DOCS[2].page_content,
}


def build_app_graph(settings: Settings | None = None) -> CompiledStateGraph:
    """Build the agent graph for the CLI (real or fake model, app corpus)."""
    settings = settings or get_settings()
    documents = load_app_documents(settings.corpus_dir)
    retriever = build_retriever(documents, embeddings=get_embeddings(settings), k=settings.top_k)
    return build_graph(get_chat_model(settings), retriever, settings)


def cmd_ask(
    question: str,
    settings: Settings | None = None,
    *,
    session_id: str = "default",
    callbacks: list | None = None,
) -> dict:
    """Ask the agent a question and return the final state dict."""
    settings = settings or get_settings()
    graph = build_app_graph(settings)
    return graph.invoke(
        {"question": question},
        config={
            "configurable": {"thread_id": session_id},
            "callbacks": callbacks if callbacks is not None else invoke_callbacks(settings),
        },
    )


def _write_sample_corpus(root: Path) -> None:
    for filename, content in _SAMPLE_CORPUS.items():
        (root / filename).write_text(content + "\n", encoding="utf-8")


def cmd_seed(settings: Settings | None = None, *, force: bool = False) -> int:
    """Ensure the corpus exists and sanity-check retrieval on it.

    The vector store is in-memory (``InMemoryVectorStore``), so "seeding"
    means preparing the corpus documents and validating that retrieval
    returns sensible results for a sample query.
    """
    settings = settings or get_settings()
    root = Path(settings.corpus_dir)
    if force or not root.is_dir() or not any(root.rglob("*")):
        root.mkdir(parents=True, exist_ok=True)
        _write_sample_corpus(root)
        print(f"Wrote sample corpus to {root}/")

    documents = load_app_documents(settings.corpus_dir)
    retriever = build_retriever(
        documents,
        embeddings=get_embeddings(settings),
        k=min(settings.top_k, max(len(documents), 1)),
    )
    print(f"Corpus: {len(documents)} document(s)")
    print("Sample retrieval for 'What is LangGraph?':")
    for doc in retriever.invoke("What is LangGraph?"):
        title = doc.metadata.get("title", "?")
        print(f"  - {title}: {doc.page_content[:60]}...")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="docsage",
        description="Agentic RAG assistant (LangChain + LangGraph + LangSmith + LangFuse).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    ask = subparsers.add_parser("ask", help="Ask the agent a question.")
    ask.add_argument("question", help="The question to ask.")
    ask.add_argument(
        "--session",
        default="default",
        help="Conversation thread id (checkpointer memory key).",
    )

    seed = subparsers.add_parser("seed", help="Prepare the corpus and sanity-check retrieval.")
    seed.add_argument(
        "--force",
        action="store_true",
        help="Rewrite the sample corpus files even if they exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    settings = get_settings()

    if args.command == "ask":
        handler = get_langfuse_handler(settings)
        callbacks = [handler] if handler else []
        result = cmd_ask(args.question, settings, session_id=args.session, callbacks=callbacks)
        print(result.get("final_answer", "(no answer)"))

        if apply_tracing_env(settings):
            print(f"[langsmith] project={settings.langchain_project} thread={args.session}")
        if apply_langfuse_env(settings) and handler is not None and handler.last_trace_id:
            url = get_langfuse_client(settings).get_trace_url(trace_id=handler.last_trace_id)
            if url:
                print(f"[langfuse] trace: {url}")
        return 0

    if args.command == "seed":
        return cmd_seed(settings, force=args.force)

    return 2


if __name__ == "__main__":
    sys.exit(main())
