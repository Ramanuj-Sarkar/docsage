"""Conditional-edge routing logic for the DocSage graph."""

from __future__ import annotations

from docsage.config import Settings
from docsage.graph.state import AgentState

ROUTE_GENERATE = "generate"
ROUTE_REWRITE = "rewrite"


def should_continue(state: AgentState, settings: Settings) -> str:
    """Route after grading: generate on relevant documents, else rewrite.

    Rewriting is bounded by ``settings.max_retries``; once the cap is reached
    we give up and generate with whatever documents we have (possibly none).
    """
    if state.get("grade_decision") == ROUTE_GENERATE:
        return ROUTE_GENERATE
    if state.get("attempts", 0) >= settings.max_retries:
        return ROUTE_GENERATE
    return ROUTE_REWRITE
