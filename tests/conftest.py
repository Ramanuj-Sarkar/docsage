"""Shared pytest fixtures for the DocSage test suite."""

from __future__ import annotations

import pytest

from docsage.config import OBSERVABILITY_ENV_VARS, Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_observability_env(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """Strip observability config so unit/graph tests never emit traces.

    This fixture is autouse: every test in the fast suite runs with LangSmith
    and LangFuse disabled, guaranteeing zero network and zero API keys. It
    removes both process env vars AND any local ``.env`` file (pydantic-settings
    would otherwise read real credentials from it).

    Integration and eval tests manage their own credentials (e.g. CI secrets),
    so they are exempt from the stripping.
    """
    if request.node.get_closest_marker("integration") or request.node.get_closest_marker("eval"):
        yield
        return
    for name in OBSERVABILITY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setitem(Settings.model_config, "env_file", None)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
