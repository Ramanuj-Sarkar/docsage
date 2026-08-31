"""Prompt templates and structured-output schemas for the DocSage nodes."""

from __future__ import annotations

from typing import Literal

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

# --- Generation (RAG answer) -------------------------------------------------

RAG_SYSTEM_PROMPT = (
    "You are DocSage, a documentation assistant. Answer the user's question "
    "using ONLY the provided context. If the context does not contain the "
    "answer, say so clearly. Cite the source document title when available."
)

rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", RAG_SYSTEM_PROMPT),
        ("human", "Context:\n{context}\n\nQuestion: {question}"),
    ]
)

# --- Relevance grading -------------------------------------------------------

GRADER_SYSTEM_PROMPT = (
    "You are a grader assessing whether a document is relevant to a user "
    "question. If the document contains keyword or semantic content related to "
    "the question, grade it as relevant. Do not be overly strict: answer 'yes' "
    "if any part of the document helps answer the question."
)


class GradeDocuments(BaseModel):
    """Structured output of the relevance grader."""

    binary_score: Literal["yes", "no"] = Field(
        description="'yes' if the document is relevant, 'no' otherwise"
    )
    explanation: str = Field(default="", description="One-sentence justification")


grade_parser = PydanticOutputParser(pydantic_object=GradeDocuments)

grade_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", GRADER_SYSTEM_PROMPT),
        (
            "human",
            "Document:\n{document}\n\nUser question: {question}\n\n{format_instructions}",
        ),
    ]
).partial(format_instructions=grade_parser.get_format_instructions())

# --- Query rewrite -----------------------------------------------------------

REWRITE_SYSTEM_PROMPT = (
    "You are a query rewriter. Rewrite the user question so it is a better, "
    "more search-friendly query for retrieving documentation. Keep it concise."
)

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", REWRITE_SYSTEM_PROMPT),
        ("human", "Original question: {question}"),
    ]
)

# --- LLM-as-judge (evaluation) -----------------------------------------------

JUDGE_SYSTEM_PROMPT = (
    "You are a strict answer judge. Decide whether the prediction answers the "
    "question the same way as the reference answer. Reply with exactly 'yes' "
    "or 'no'."
)

judge_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", JUDGE_SYSTEM_PROMPT),
        ("human", "Prediction: {prediction}\n\nReference: {reference}"),
    ]
)
