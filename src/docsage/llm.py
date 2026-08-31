"""Chat-model factory.

The factory is the single place that constructs chat models so that tests can
inject scripted fake models (``provider="fake"``) and the CLI can use real
models (``provider="openai"``). LangGraph nodes receive the model via
dependency injection (``build_graph(llm)``), never by importing it here.
"""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from docsage.config import Settings, get_settings

Provider = Literal["openai", "anthropic", "fake"]

# Providers with working implementations. "anthropic" is planned but requires
# installing ``langchain-anthropic`` (kept out of Phase 1 to stay lean).
SUPPORTED_PROVIDERS: tuple[str, ...] = ("openai", "fake")

# Canned JSON grade reply for the fake provider (matches the GradeDocuments schema).
FAKE_GRADE_YES = '{"binary_score": "yes", "explanation": "fake offline model"}'

# Markers used to route fake replies to the right chain (see ScriptedChatModel).
FAKE_GRADE_MARKER = "grader assessing"
FAKE_REWRITE_MARKER = "query rewriter"


class ScriptedChatModel(BaseChatModel):
    """Chat model that routes replies by marker substrings in the prompt text.

    ``responses`` maps a marker substring to a sequence of replies cycled per
    marker. Used as the ``fake`` provider so the CLI runs fully offline, and by
    tests to script per-role behaviors (e.g. a "no-then-yes" grading sequence).
    """

    responses: dict[str, list[str]]
    fallback: str = "This is an offline demo answer."
    model_name: str = "fake-offline"
    _index: dict[str, int] = PrivateAttr(default_factory=dict)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object | None = None,
        **kwargs: object,
    ) -> ChatResult:
        prompt_text = " ".join(str(message.content) for message in messages)
        for marker, replies in self.responses.items():
            if marker in prompt_text:
                position = self._index.get(marker, 0)
                reply = replies[position % len(replies)]
                self._index[marker] = position + 1
                return ChatResult(generations=[ChatGeneration(message=AIMessage(content=reply))])
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=self.fallback))])


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    """Return a chat model for the configured provider.

    - ``"fake"``: deterministic, marker-routed model — runs the whole graph
      offline (demo/CI). Tests can build their own ``ScriptedChatModel``.
    - ``"openai"``: :class:`~langchain_openai.ChatOpenAI`.
    - ``"anthropic"``: not installed yet; raises a helpful error.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider

    if provider == "fake":
        return ScriptedChatModel(
            responses={
                FAKE_GRADE_MARKER: [FAKE_GRADE_YES],
                FAKE_REWRITE_MARKER: ["rewritten query"],
            }
        )

    if provider == "openai":
        # Deferred import: keeps the module importable without the OpenAI SDK.
        from langchain_openai import ChatOpenAI

        kwargs: dict[str, object] = {"model": settings.llm_model, "temperature": 0}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        return ChatOpenAI(**kwargs)

    raise ValueError(
        f"Unsupported LLM provider {provider!r}. Supported: "
        f"{', '.join(SUPPORTED_PROVIDERS)}. 'anthropic' requires installing "
        "langchain-anthropic (planned)."
    )
