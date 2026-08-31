"""Tests for agent tools."""

from __future__ import annotations

from datetime import date

import pytest

from docsage.tools import calculator, today


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("1 + 2", "3"),
        ("(3 + 5) * 2", "16"),
        ("10 / 4", "2.5"),
        ("7 // 2", "3"),
        ("2 ** 10", "1024"),
        ("-3 + 5", "2"),
        ("1.5 * 2", "3.0"),
    ],
)
def test_calculator_arithmetic(expression: str, expected: str) -> None:
    assert calculator.invoke({"expression": expression}) == expected


def test_calculator_invalid_syntax_returns_error() -> None:
    assert calculator.invoke({"expression": "1 +"}).startswith("Error:")


def test_calculator_division_by_zero_returns_error() -> None:
    assert calculator.invoke({"expression": "1 / 0"}).startswith("Error:")


def test_calculator_rejects_code_execution() -> None:
    # Names, calls, and attribute access are blocked — no code can run.
    malicious = "__import__('os').system('echo pwned')"
    assert calculator.invoke({"expression": malicious}).startswith("Error:")


@pytest.mark.parametrize(
    "expression",
    [
        "x + 1",  # name lookup is blocked
        "1 < 2",  # comparison operators are blocked
        "1 & 2",  # bitwise operators are blocked
        "not True",  # boolean constants/operators are blocked
        "'text'",  # non-numeric constants are blocked
    ],
)
def test_calculator_rejects_unsupported_syntax(expression: str) -> None:
    assert calculator.invoke({"expression": expression}).startswith("Error:")


def test_today_returns_iso_date() -> None:
    assert today.invoke({}) == date.today().isoformat()
