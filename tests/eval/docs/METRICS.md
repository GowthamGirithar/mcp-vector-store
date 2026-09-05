# Hybrid Search - Retrieval - eval

## Out of scope

This eval covers **retrieval only** — the `hybrid_search` tool. It does not
cover the generation part of a RAG pipeline (no answer is generated from the
retrieved chunks, and no generation quality is scored).

This is why **answer relevancy** and **faithfulness** are not covered: both
are Ragas metrics that score a *generated* answer against the question
and/or the retrieved context. With no generation step in this eval, there is
no generated answer to score them against.

## Metrics

| Metric | Definition |
|---|---|
| `hit_rate@k` | Deterministic. 1.0 if *any* chunk `hybrid_search` returns matches *any* of the question's ground-truth chunks, else 0.0. Averaged over all non-negative questions. |
| `mrr` | Deterministic. `1 / rank` of the first matching chunk (0 if none matched). Averaged over all non-negative questions. |
| `ndcg` | Deterministic. Binary-relevance NDCG@k: `DCG = sum(1/log2(rank+1))` over every retrieved chunk that's a ground-truth chunk, normalized by the ideal DCG (all ground-truth chunks ranked first). Averaged over all non-negative questions. |
| `context_precision` | Ragas `LLMContextPrecisionWithReference`, OpenAI-judged. For each retrieved chunk, is it actually relevant to answering the question, weighted by rank. |
| `context_recall` | Ragas `LLMContextRecall`, OpenAI-judged. Decomposes the ground-truth answer into individual claims and checks whether each is supported by the retrieved chunks combined. |

`hit_rate`/`mrr`/`ndcg` are computed against `ground_truth_chunk_indices`
(structural ground truth: which chunks *should* come back) and require no
LLM call. `context_precision`/`context_recall` are computed against
`ground_truth_answer` (semantic ground truth: what the answer actually
says), judged by an LLM. They deliberately check different things — a
question can score a perfect `hit_rate` while still scoring low on
`context_recall` if only part of a multi-hop answer was retrieved.

Negative-control questions (`metadata.difficulty: "negative"`) are excluded
from all aggregates — there's no ground-truth chunk or answer to score
against. They're included in the golden set to eyeball whether
`hybrid_search` confidently returns something that *looks* like an answer
for a question the paper never addresses.

## Golden Dataset

**Source:** `tests/fixtures/attention.pdf` (the "Attention Is All You Need"
paper). Questions were hand-authored against this single document.

**Where it's maintained:**

- `tests/eval/datasets/attention_qa.jsonl` — the golden Q&A set itself. Each
  question references ground truth by **`chunk_index`**, not `chunk_id`:
  `chunk_id`s are regenerated on every ingestion (see
  `tools/document_embedding.py`), so `chunk_index` (deterministic parse
  order) is the only key that survives re-ingestion. Non-scoring fields
  (e.g. `difficulty`) live under a nested `metadata` object rather than as
  top-level keys, so the record's scoring inputs stay unambiguous as more
  metadata gets added.
- `tests/eval/datasets/attention_chunks.json` — a point-in-time dump of
  every chunk's text, kept for reference when authoring new questions
  against known chunk boundaries/content. Regenerate manually via
  `generate_document_embedding(force=True)` +
  `vector_db.get_all_documents(...)` if the golden set needs rebuilding from
  scratch (the one-off script used to produce it was removed after initial
  authoring).

**What's out of scope for this dataset:**

- Table/image chunks are excluded entirely — no question's ground truth
  touches a `has_table`/`has_image` chunk.
- Only one source document (`attention.pdf`) is covered — no
  cross-document or multi-document retrieval questions.
- 20 scored questions (+3 negative controls) is enough to sanity-check the
  eval harness, not a large enough set to treat any single-point score
  change as statistically meaningful.
