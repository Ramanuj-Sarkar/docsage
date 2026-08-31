"""LangChain tools exposed to the agent."""

from __future__ import annotations

import ast
import operator
from datetime import date

from langchain_core.tools import BaseTool, tool

_ALLOWED_BINOPS: dict[type[ast.operator], object] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_ALLOWED_UNARYOPS: dict[type[ast.unaryop], object] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _eval_expr(node: ast.AST) -> float:
    """Evaluate an arithmetic AST safely (no names, calls, or attribute access)."""
    if isinstance(node, ast.Expression):
        return _eval_expr(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")
    if isinstance(node, ast.BinOp):
        op = _ALLOWED_BINOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_expr(node.left), _eval_expr(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _ALLOWED_UNARYOPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        return op(_eval_expr(node.operand))
    raise ValueError(f"Unsupported syntax: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression.

    Supports ``+ - * / // % **``, parentheses, and numbers only.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _eval_expr(tree)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError) as exc:
        return f"Error: {exc}"
    return str(result)


@tool
def today() -> str:
    """Return today's date as YYYY-MM-DD."""
    return date.today().isoformat()


AGENT_TOOLS: list[BaseTool] = [calculator, today]
