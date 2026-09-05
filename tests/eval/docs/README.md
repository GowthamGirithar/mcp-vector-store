# Eval

Offline retrieval-quality eval for the `hybrid_search` tool, against a
hand-built golden Q&A set sourced from `tests/fixtures/attention.pdf`.

## How to run

```bash
make install-eval          # once, pulls in ragas/langchain on top of requirements.txt
export OPENAI_API_KEY=...  # or set it in .env — needed for the judge only

make eval-deterministic     # hit_rate@k / MRR / NDCG only — no OpenAI calls, fast, free
make eval-all               # adds Ragas context_precision / context_recall, judged by OpenAI
```

## Metrics

See [`METRICS.md`](METRICS.md) — what's covered/out of scope, the metrics
definitions, and how the golden dataset is prepared and maintained.

## Results

Each run writes a timestamped report (`.md` + `.json`) to
`tests/eval/experiments/` — gitignored, since these are run artifacts, not
source.
