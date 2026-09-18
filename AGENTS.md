# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project

An MCP (Model Context Protocol) server providing vector database
functionality: text/document storage with automatic embedding, similarity
search, hybrid (vector + BM25) search with RRF fusion and optional
cross-encoder reranking, and multimodal document ingestion. See `README.md`
for the full tool catalog and architecture diagrams.

## Setup

```bash
pip install -r requirements.txt            # runtime deps
pip install -r requirements-dev.txt        # + testing/linting
pip install -r requirements-eval.txt       # + ragas/langchain, for evals
cp .env.example .env                       # then edit .env
```

System deps: `tesseract`, `poppler` (macOS: `brew install tesseract poppler`) —
required by `unstructured` for document parsing.

## Run

```bash
python main.py          # or: make run
```

Transport mode (`stdio` / `sse` / `streamable-http`) is set via `MCP_TRANSPORT` in `.env`.

## Test

```bash
make test              # full pytest suite (unit + integration)
make test-unit          # mocked, fast — no real vector DB / embedding service
make test-integration    # against real (temp-dir) vector DB + embedding services
make eval-deterministic # hit-rate/MRR/NDCG only, no OpenAI judge
make eval-all           # + Ragas context_precision/context_recall via OpenAI judge
```

Always run `make test-unit` after a change; run `make test-integration` and
the relevant eval target when touching retrieval/ranking behavior.

## Package layout (`src/mcp_vectordb/`)

- `adapters/` — vector DB adapter interface + ChromaDB implementation + factory
- `chunking/` — document parsing/chunking
- `core/` — document embedding orchestration (fixed-size and LLM-driven "agentic" chunking)
- `embedding/` — embedding provider abstraction (OpenAI, Sentence-Transformers), cache, compression (`quantization.py`)
- `llm/` — LLM completion service abstraction, used by agentic embedding
- `search/` — BM25 index, RRF fusion, cross-encoder reranker, TurboQuant scan, shadow-mode comparison
- `models/` — shared data models (`Document`)
- `tools/` — MCP tool definitions (`storage.py`, `search.py`, `document_embedding.py`, `agentic_embedding.py`)
- `config/` — environment-driven settings (`config.py`, reads `.env`)

## Conventions

- **No comments unless the WHY is non-obvious.** Well-named identifiers
  cover the what. Comments in this codebase explain hidden constraints,
  invariants, or the reasoning behind a non-default choice — see
  `search/shadow.py` or `adapters/chroma.py` for the expected density/style.
- **New optional behavior is a config flag, off by default.** Follow the
  `CompressionConfig` / `SearchConfig` pattern in `config/config.py`: a
  `bool` field defaulting to `False`, read from an env var in
  `Settings.from_env`, documented in `.env.example`.
- **Tracing is optional and must degrade silently.** Modules that use
  LangSmith's `@traceable` import it inside a `try/except ImportError` with
  a no-op decorator fallback (see `search/shadow.py`,
  `search/turboquant_scan.py`). Don't make `langsmith` a hard dependency of
  any new traced module.
- **MCP tools are declared with `@mcp.tool()`** in `tools/*.py`, thin
  wrappers around an inner `@traceable`-decorated `_*_traced` function that
  holds the real logic — keep that split when adding a tool.
- **Background/shadow work must never affect the real response path.**
  `search/shadow.py` fires via `asyncio.create_task` (fire-and-forget) and
  swallows all exceptions with `logger.exception`, never raising. Follow
  this pattern for any other non-blocking evaluation/comparison work.
- **Feature flags that change stored data need a migration story, not just
  a flip.** Enabling `compression.enabled` does not retroactively encode
  already-stored documents — see `docs/turboquant_and_shadow.md` for the
  reasoning; apply the same thinking to any new flag that changes what gets
  written per document.

## Docs

- `README.md` — tool catalog, architecture diagrams, how to run/connect.
- `docs/turboquant_and_shadow.md` — TurboQuant compression + shadow-mode
  evaluation design, current limitations, and open investigation items for
  moving compressed codes from Chroma metadata into a real vector-indexed
  store.
