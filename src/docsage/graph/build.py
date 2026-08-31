"""Compile the DocSage LangGraph agent."""

from __future__ import annotations

from functools import partial

from langchain_core.language_models import BaseChatModel
from langchain_core.runnables import Runnable
from langchain_core.vectorstores import VectorStoreRetriever
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from docsage.config import Settings, get_settings
from docsage.graph.edges import should_continue
from docsage.graph.nodes import finalize, generate, grade, retrieve, rewrite
from docsage.graph.state import AgentState
from docsage.prompts import grade_parser, grade_prompt, rag_prompt, rewrite_prompt


def _build_chains(llm: BaseChatModel) -> dict[str, Runnable]:
    """Build the LLM chains used by the nodes (one LLM instance per chain)."""
    return {
        "grade": grade_prompt | llm | grade_parser,
        "rewrite": rewrite_prompt | llm,
        "generate": rag_prompt | llm,
    }


def _build_state_graph(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    settings: Settings,
) -> StateGraph:
    """Assemble the graph topology (uncompiled, inspectable by tests)."""
    chains = _build_chains(llm)

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", partial(retrieve, retriever=retriever))
    graph.add_node("grade", partial(grade, chain=chains["grade"]))
    graph.add_node("rewrite", partial(rewrite, chain=chains["rewrite"]))
    graph.add_node("generate", partial(generate, chain=chains["generate"]))
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", partial(should_continue, settings=settings))
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", "finalize")
    graph.add_edge("finalize", END)
    return graph


def build_graph(
    llm: BaseChatModel,
    retriever: VectorStoreRetriever,
    settings: Settings | None = None,
) -> CompiledStateGraph:
    """Compile the agent graph with an in-memory checkpointer."""
    settings = settings or get_settings()
    graph = _build_state_graph(llm, retriever, settings)
    return graph.compile(checkpointer=InMemorySaver())
