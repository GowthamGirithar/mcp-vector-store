# hybrid_search retrieval eval

Offline retrieval-quality eval for the `hybrid_search` tool, against a
hand-built golden Q&A set sourced from `tests/fixtures/attention.pdf`.
Scope of this first pass: text-only questions (no table/image ground truth),
`hybrid_search` only (not `generate_document_embedding` or
`similarity_search`).

## Setup

```bash
.venv/bin/pip install -r requirements-eval.txt
export OPENAI_API_KEY=...   # or set it in .env — needed for the judge only
```

## Run

```bash
# Deterministic hit-rate@k / MRR only (no OpenAI calls, fast, free)
PYTHONPATH=src .venv/bin/python -m tests.eval.run_eval --no-judge

# Full run: adds Ragas context_precision / context_recall, judged by OpenAI
PYTHONPATH=src .venv/bin/python -m tests.eval.run_eval
```

Each run writes a timestamped report to `tests/eval/reports/` (`.md` +
`.json`; gitignored — these are run artifacts, not source).

## How it fits together

- `golden/attention_qa.jsonl` — the golden set. Each question references
  ground truth by **`chunk_index`**, not `chunk_id`: `chunk_id`s are
  regenerated on every ingestion (see `tools/document_embedding.py`), so
  `chunk_index` (deterministic parse order) is the only key that survives
  re-ingestion. `golden/attention_chunks.json` is a point-in-time dump of
  every chunk's text, kept for reference when writing new questions.
- `setup_services.py` — boots a real Chroma adapter + local
  (sentence-transformers) embedding service as `services` module globals,
  the same way `tests/conftest.py`'s `real_services` fixture does for unit
  tests, so `hybrid_search` can be called directly without a running FastMCP
  server. Uses a dedicated on-disk path (`tests/eval/.eval_chroma_db/`,
  gitignored) so eval runs never touch the project's real `chroma_db/`.
- `ingest.py` — one-off script: ingests `attention.pdf` into the eval
  collection and dumps every chunk to `golden/attention_chunks.json`. Re-run
  it (it uses `force=True`) only when re-authoring the golden set from
  scratch — every existing question's `chunk_index` stays valid across
  re-ingestion since chunking is deterministic, but chunk_ids will churn.
- `run_eval.py` — the actual eval: ingests (idempotently, `force=False`) if
  needed, calls `hybrid_search` for each golden question, and scores it two
  ways:
  - **Deterministic**: hit-rate@k (did any returned chunk match a ground
    truth `chunk_index`) and MRR — free, catches harness bugs before
    spending judge calls.
  - **Ragas + OpenAI judge** (`judge.py`, default `gpt-4o-mini`):
    `context_precision` and `context_recall`, scored against each
    question's `ground_truth_answer` — retrieval-only, no answer-generation
    step needed.
- Negative-control questions (`difficulty: "negative"`, empty
  `ground_truth_chunk_indices`) are excluded from the aggregate scores —
  there's no ground truth to score against — and are listed separately in
  the report for manual review of what `hybrid_search` confidently returned
  for an unanswerable question.

## Results

See `RESULTS.md` for the metrics glossary and the run-by-run results log
(per-question scores, findings, known gaps) — kept separate from this file
so it doesn't get stale relative to the how-to instructions above.
