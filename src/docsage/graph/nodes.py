"""Node implementations for the DocSage LangGraph agent.

Nodes are plain functions of ``(state, **deps)`` so they can be unit-tested
directly; :func:`docsage.graph.build.build_graph` wraps them with
``functools.partial`` to inject the LLM chains, retriever, and settings.
"""

from __future__ import annotations

import logging

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever

from docsage.graph.edges import ROUTE_GENERATE, ROUTE_REWRITE
from docsage.graph.state import AgentState

logger = logging.getLogger(__name__)


def _format_context(documents: list[Document]) -> str:
    parts = []
    for index, doc in enumerate(documents, start=1):
        title = doc.metadata.get("title", f"source-{index}")
        parts.append(f"[{index}] ({title}) {doc.page_content}")
    return "\n\n".join(parts)


def retrieve(state: AgentState, retriever: VectorStoreRetriever) -> dict:
    """Retrieve the top-k documents for the current search query."""
    query = state.get("search_query") or state["question"]
    return {"documents": retriever.invoke(query)}


def grade(state: AgentState, chain: Runnable) -> dict:
    """Grade every retrieved document and keep only the relevant ones.

    Returns the routing decision in ``grade_decision`` and replaces
    ``documents`` with the relevant subset.
    """
    documents = state.get("documents", [])
    if not documents:
        return {"grade_decision": ROUTE_REWRITE, "documents": []}

    relevant: list[Document] = []
    for doc in documents:
        try:
            parsed = chain.invoke({"document": doc.page_content, "question": state["question"]})
        except OutputParserException:
            # The grader LLM sometimes fails to produce valid JSON; treat the
            # document as irrelevant rather than crashing the whole run. The
            # rewrite loop is bounded by the retry cap, so this cannot loop.
            logger.warning(
                "Grader output unparseable for document %r; treating as irrelevant",
                doc.page_content[:60],
            )
            continue
        if parsed.binary_score == "yes":
            relevant.append(doc)

    decision = ROUTE_GENERATE if relevant else ROUTE_REWRITE
    return {"grade_decision": decision, "documents": relevant}


def rewrite(state: AgentState, chain: Runnable) -> dict:
    """Rewrite the search query to improve retrieval on the next pass."""
    response = chain.invoke({"question": state["question"]})
    return {
        "search_query": response.content,
        "attempts": state.get("attempts", 0) + 1,
    }


def generate(state: AgentState, chain: Runnable) -> dict:
    """Generate an answer from the retrieved (relevant) documents."""
    context = _format_context(state.get("documents", []))
    response = chain.invoke({"context": context, "question": state["question"]})
    return {"generation": response.content}


def finalize(state: AgentState) -> dict:
    """Post-process: promote the generation to the final answer."""
    return {"final_answer": state.get("generation", "")}
