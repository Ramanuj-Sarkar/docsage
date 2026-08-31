"""Tests for the DocSage CLI (offline, fake provider)."""

from __future__ import annotations

from docsage.config import Settings
from docsage.main import build_app_graph, build_parser, cmd_ask, cmd_seed


def _settings(tmp_path) -> Settings:
    return Settings(llm_provider="fake", corpus_dir=str(tmp_path / "corpus"))


def test_cmd_ask_returns_final_answer(tmp_path) -> None:
    result = cmd_ask("What is LangGraph?", _settings(tmp_path))
    assert result.get("final_answer") == "This is an offline demo answer."


def test_cmd_ask_routes_to_generate_offline(tmp_path) -> None:
    result = cmd_ask("What is LangGraph?", _settings(tmp_path))
    assert result.get("grade_decision") == "generate"
    assert len(result.get("documents", [])) == 3  # built-in fixture corpus


def test_cmd_ask_uses_provided_callbacks(tmp_path) -> None:
    sentinel: list = []
    result = cmd_ask(
        "What is LangGraph?",
        _settings(tmp_path),
        callbacks=sentinel,
    )
    assert result.get("final_answer")


def test_cmd_seed_creates_sample_corpus(tmp_path) -> None:
    settings = _settings(tmp_path)
    assert cmd_seed(settings) == 0
    files = {
        (tmp_path / "corpus").joinpath(name)
        for name in ("langgraph.md", "langsmith.md", "langfuse.md")
    }
    assert all(path.is_file() for path in files)


def test_cmd_seed_idempotent_without_force(tmp_path) -> None:
    settings = _settings(tmp_path)
    assert cmd_seed(settings) == 0
    corpus = tmp_path / "corpus"
    (corpus / "extra.md").write_text("extra", encoding="utf-8")
    assert cmd_seed(settings) == 0  # no rewrite, extra.md preserved
    assert (corpus / "extra.md").is_file()


def test_build_app_graph_compiles(tmp_path) -> None:
    graph = build_app_graph(_settings(tmp_path))
    nodes = set(graph.get_graph().nodes)
    assert {"retrieve", "grade", "rewrite", "generate", "finalize"} <= nodes


def test_build_parser_accepts_ask_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["ask", "hello", "--session", "s1"])
    assert args.command == "ask"
    assert args.question == "hello"
    assert args.session == "s1"


def test_build_parser_accepts_seed_subcommand() -> None:
    parser = build_parser()
    args = parser.parse_args(["seed", "--force"])
    assert args.command == "seed"
    assert args.force is True
