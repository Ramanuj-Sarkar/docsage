"""Application settings for DocSage.

Values are read from environment variables or a local ``.env`` file
(pydantic-settings). This module is the single source of truth for
configuration; ``observability.py`` builds LangSmith/LangFuse clients from it.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Environment variable names that control observability; used by tests to
# guarantee a clean (offline) environment.
OBSERVABILITY_ENV_VARS: tuple[str, ...] = (
    "LANGCHAIN_TRACING_V2",
    "LANGCHAIN_API_KEY",
    "LANGCHAIN_PROJECT",
    "LANGCHAIN_ENDPOINT",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
)


class Settings(BaseSettings):
    """Runtime configuration for DocSage."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Model provider ---------------------------------------------------
    openai_api_key: str | None = Field(default=None, description="OpenAI API key.")
    anthropic_api_key: str | None = Field(default=None, description="Anthropic API key.")
    llm_provider: str = Field(
        default="openai",
        description="Model provider: 'openai', 'anthropic', or 'fake' (tests).",
    )
    llm_model: str = Field(default="gpt-4o-mini", description="Model name to use.")

    # --- LangSmith --------------------------------------------------------
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangSmith auto-tracing.")
    langchain_api_key: str | None = Field(default=None, description="LangSmith API key.")
    langchain_project: str = Field(default="docsage", description="LangSmith project name.")
    langchain_endpoint: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith tracing endpoint.",
    )

    # --- LangFuse ---------------------------------------------------------
    langfuse_public_key: str | None = Field(default=None, description="LangFuse public key.")
    langfuse_secret_key: str | None = Field(default=None, description="LangFuse secret key.")
    langfuse_host: str = Field(default="https://cloud.langfuse.com", description="LangFuse host.")

    # --- App behaviour ----------------------------------------------------
    max_retries: int = Field(
        default=2,
        ge=1,
        description="Max query-rewrite attempts in the graph.",
    )
    vector_store_path: str = Field(
        default=".data/vectorstore",
        description="Vector store snapshot path.",
    )
    corpus_dir: str = Field(
        default="corpus",
        description="Directory of .md/.txt documents used for retrieval.",
    )
    top_k: int = Field(
        default=4,
        ge=1,
        description="Number of documents retrieved per query.",
    )

    @property
    def tracing_enabled(self) -> bool:
        """True when any observability backend is configured."""
        return self.langchain_tracing_v2 or bool(self.langfuse_public_key)


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings (computed once per process)."""
    return Settings()
