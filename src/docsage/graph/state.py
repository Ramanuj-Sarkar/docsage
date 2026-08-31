"""State schema shared by all LangGraph nodes."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.documents import Document


class AgentState(TypedDict):
    """State passed between LangGraph nodes.

    Fields are written by exactly one node at a time, so plain (last-write-wins)
    semantics are sufficient — no reducers needed.
    """

    question: str
    search_query: str
    documents: list[Document]
    grade_decision: str
    generation: str
    attempts: int
    final_answer: str
