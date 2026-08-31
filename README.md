# DocSage

Agentic RAG documentation assistant built on **LangChain**, **LangGraph**,
**LangSmith**, and **LangFuse** — tested with **pytest**.

> **Status: complete (Phases 0–5).** All suites pass: 95 fast tests run fully
> offline; `pytest -m integration` and `pytest -m eval` pass against live
> LangSmith/LangFuse. One real run traces to both platforms.

## Architecture

```
 user question
      │
      ▼
┌─ LangGraph StateGraph (InMemorySaver checkpointer) ─────────────┐
│  retrieve → grade → relevant? → yes → generate → finalize        │
│                      │ no                        ▲                │
│                      ▼                           │                │
│                    rewrite ──────(retry cap)─────┘                │
└───────────────┬──────────────────────────────────────────────────┘
                │
   ┌────────────▼────────────┐   ┌──────────────────────────────────┐
   │ LangSmith (auto-trace    │   │ LangFuse (LangfuseCallbackHandler│
   │ via env vars, datasets,  │   │  attached per invoke, scores)    │
   │ evaluate())              │   │                                  │
   └─────────────────────────┘   └──────────────────────────────────┘
```

- **LangChain** — model factory (openai / fake), prompts, output parsers,
  `InMemoryVectorStore` retrieval, tools (safe AST calculator, date).
- **LangGraph** — stateful agent: `retrieve → grade → (rewrite ↺ | generate) →
  finalize`, conditional routing bounded by `DOCSAGE_MAX_RETRIES`.
- **LangSmith** — auto-tracing via `apply_tracing_env()`, dataset evals with
  exact-match / contains-reference / LLM-judge evaluators.
- **LangFuse** — `CallbackHandler` per invoke (`invoke_callbacks()`), trace
  capture, `create_score` scoring.
- **pytest** — 4 layers: unit (no network), graph (scripted model), integration
  (real API), eval (datasets on both platforms).

## Quick start

```bash
python -m venv .venv
.venv/bin/pip install -e ".[dev]"
cp .env.example .env          # fill in keys (only needed for live suites)
.venv/bin/docsage seed        # writes the sample corpus/ (offline OK)
.venv/bin/docsage ask "What is LangGraph?"   # offline demo (LLM_PROVIDER=fake)
```

## CLI

```bash
docsage seed [--force]            # prepare corpus + retrieval sanity check
docsage ask "QUESTION" [--session ID]   # run the agent (threaded memory)
```

With real keys in `.env`, `ask` prints the answer plus LangSmith project and
LangFuse trace URL. With `LLM_PROVIDER=fake` the whole graph runs offline.

## Testing

```bash
.venv/bin/pytest                          # fast suite: unit + graph, no network
.venv/bin/pytest -m integration           # real LLM + both observability platforms
.venv/bin/pytest -m eval                  # LangSmith dataset eval + LangFuse scoring
.venv/bin/python scripts/run_evals.py     # both live suites in one go
.venv/bin/python scripts/seed_vectorstore.py
```

The fast suite (95 tests, ~92% coverage, gate 85%) is hardened to run with
**zero network and zero API keys**: a conftest fixture strips observability env
vars *and* any local `.env`, and LLM calls are replaced by the scripted
`ScriptedChatModel`. Live suites skip with a clear message when credentials are
missing.

## Configuration (`.env`)

| Variable                                                         | Purpose                             |
|------------------------------------------------------------------|-------------------------------------|
| `OPENAI_API_KEY`, `LLM_PROVIDER`, `LLM_MODEL`                    | Model provider (`openai` or `fake`) |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` | LangSmith tracing                   |
| `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`    | LangFuse                            |
| `DOCSAGE_MAX_RETRIES`, `DOCSAGE_TOP_K`, `DOCSAGE_CORPUS_DIR`     | Agent behaviour                     |

## Layout

- `src/docsage/` — `config`, `llm`, `prompts`, `embeddings`, `retrieval`,
  `tools`, `documents`, `graph/` (state, nodes, edges, build), `observability`,
  `evals`, `datasets`, `main` (CLI)
- `tests/` — `unit/`, `graph/`, `integration/`, `eval/`
- `datasets/qa_pairs.jsonl` — shared eval dataset (both platforms)
- `scripts/` — `seed_vectorstore.py`, `run_evals.py`
- `.github/workflows/ci.yml` — fast CI + nightly eval job (secrets)

## Notes / known limitations

- Vector store is `InMemoryVectorStore` (pure Python; no Chroma/FAISS on
  Python 3.14). Swap `build_retriever` internals for pgvector/Qdrant for
  production persistence — the signature stays the same.
- LangFuse cloud ingestion can take ~40s; live tests poll for up to a minute.
- `list_runs()` in langsmith is deprecated (removal Jan 2027) in favor of
  `client.runs.query()`; kept until the new API stabilizes.
