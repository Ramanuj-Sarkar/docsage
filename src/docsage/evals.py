"""Evaluators for LangSmith (and later LangFuse) dataset evals.

Each evaluator is a plain ``(run, example) -> EvaluationResult`` function as
expected by :func:`langsmith.evaluation.evaluate`, which keeps them
unit-testable offline with stub run/example objects.
"""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.runnables import Runnable
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run


def _prediction_text(run: Run) -> str:
    outputs = run.outputs or {}
    if isinstance(outputs, dict):
        for key in ("answer", "output"):
            if outputs.get(key):
                return str(outputs[key])
        return ""
    return str(outputs)


def exact_match_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Score 1.0 when the prediction equals the reference answer, else 0.0."""
    reference = str(example.outputs.get("answer", ""))
    return EvaluationResult(
        key="exact_match",
        score=float(_prediction_text(run).strip() == reference.strip()),
    )


def contains_reference_evaluator(run: Run, example: Example) -> EvaluationResult:
    """Score 1.0 when the prediction contains the reference answer text."""
    reference = str(example.outputs.get("answer", ""))
    return EvaluationResult(
        key="contains_reference",
        score=float(reference.strip() in _prediction_text(run)),
    )


def llm_judge_evaluator(
    judge_chain: Runnable,
) -> Callable[[Run, Example], EvaluationResult]:
    """Factory for an LLM-as-judge evaluator.

    ``judge_chain`` must accept ``{"prediction", "reference"}`` and reply with
    an AIMessage whose content is exactly ``"yes"`` (matches) or ``"no"``.
    """

    def _judge(run: Run, example: Example) -> EvaluationResult:
        reference = str(example.outputs.get("answer", ""))
        response = judge_chain.invoke({"prediction": _prediction_text(run), "reference": reference})
        verdict = str(response.content).strip().lower()
        return EvaluationResult(
            key="llm_judge",
            score=float(verdict == "yes"),
        )

    return _judge
