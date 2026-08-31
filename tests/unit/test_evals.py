"""Tests for LangSmith evaluators (offline, with stub run/example objects)."""

from __future__ import annotations

from types import SimpleNamespace

from docsage.evals import (
    contains_reference_evaluator,
    exact_match_evaluator,
    llm_judge_evaluator,
)


def _run(answer: str) -> SimpleNamespace:
    return SimpleNamespace(outputs={"answer": answer})


def _example(answer: str) -> SimpleNamespace:
    return SimpleNamespace(outputs={"answer": answer})


class _JudgeChain:
    """Stub judge chain: returns a fixed verdict and records its inputs."""

    def __init__(self, verdict: str) -> None:
        self.verdict = verdict
        self.last_input: dict | None = None

    def invoke(self, inputs: dict, config: object | None = None) -> SimpleNamespace:
        self.last_input = inputs
        return SimpleNamespace(content=self.verdict)


def test_exact_match_scores_1_on_identical() -> None:
    result = exact_match_evaluator(_run("LangGraph"), _example("LangGraph"))
    assert result.key == "exact_match"
    assert result.score == 1.0


def test_exact_match_scores_0_on_different() -> None:
    result = exact_match_evaluator(_run("LangSmith"), _example("LangGraph"))
    assert result.score == 0.0


def test_exact_match_ignores_whitespace() -> None:
    result = exact_match_evaluator(_run("  LangGraph  "), _example("LangGraph"))
    assert result.score == 1.0


def test_contains_reference_scores_1_on_substring() -> None:
    result = contains_reference_evaluator(
        _run("LangGraph is a graph library"), _example("LangGraph")
    )
    assert result.score == 1.0


def test_contains_reference_scores_0_on_missing() -> None:
    result = contains_reference_evaluator(_run("nothing here"), _example("LangGraph"))
    assert result.score == 0.0


def test_llm_judge_scores_yes() -> None:
    chain = _JudgeChain("yes")
    evaluator = llm_judge_evaluator(chain)
    result = evaluator(_run("LangGraph"), _example("LangGraph"))
    assert result.key == "llm_judge"
    assert result.score == 1.0
    assert chain.last_input == {"prediction": "LangGraph", "reference": "LangGraph"}


def test_llm_judge_scores_no() -> None:
    chain = _JudgeChain("no")
    result = llm_judge_evaluator(chain)(_run("wrong"), _example("LangGraph"))
    assert result.score == 0.0


def test_prediction_falls_back_to_output_key() -> None:
    run = SimpleNamespace(outputs={"output": "LangGraph"})
    result = exact_match_evaluator(run, _example("LangGraph"))
    assert result.score == 1.0


def test_prediction_empty_when_no_known_output_key() -> None:
    run = SimpleNamespace(outputs={"other": "LangGraph"})
    result = exact_match_evaluator(run, _example("LangGraph"))
    assert result.score == 0.0
