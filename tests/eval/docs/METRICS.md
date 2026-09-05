# hybrid_search retrieval eval — results log

Reference doc for what the metrics mean and a running log of eval runs, kept
separate from `README.md` (which covers setup/how-to). See `README.md` for
how to reproduce any of these numbers.

## Metrics glossary

| Metric | Computed how | What it actually catches |
|---|---|---|
| `hit_rate@k` | 1.0 if *any* chunk `hybrid_search` returns matches *any* of the question's `ground_truth_chunk_indices`, else 0.0. Averaged over all non-negative questions. | Coarse recall — retrieval found at least one relevant chunk somewhere in the top-k. Cheap (no LLM call), but blind to whether *all* the info needed to answer was retrieved. |
| `mrr` | `1 / rank` of the first matching chunk (0 if none matched). Averaged over all non-negative questions. | Ranking quality — rewards a relevant chunk appearing earlier in the returned list, not just being present somewhere in it. |
| `context_precision` | Ragas `LLMContextPrecisionWithReference`, OpenAI-judged: for each retrieved chunk, is it actually relevant to answering the question, weighted by rank. | Noise in the returned set — a high hit-rate with low precision means retrieval is surfacing correct-and-irrelevant chunks together. |
| `context_recall` | Ragas `LLMContextRecall`, OpenAI-judged: decomposes `ground_truth_answer` into individual claims and checks whether each is supported by the retrieved chunks *combined*. | Full-answer coverage — catches the case `hit_rate` misses: a multi-hop question where only part of the needed information was retrieved. |

`hit_rate`/`mrr` are computed against `ground_truth_chunk_indices` (structural
ground truth: which chunks *should* come back). `context_precision`/`recall`
are computed against `ground_truth_answer` (semantic ground truth: what the
answer actually says), judged by an LLM. They're deliberately checking
different things — see the q19 disagreement below for why that matters.

Negative-control questions (`difficulty: "negative"`) are excluded from all
aggregates — there's no ground-truth chunk or answer to score against. They're
included in the golden set to eyeball whether `hybrid_search` confidently
returns something that *looks* like an answer for a question the paper never
addresses.

## Runs

### 2026-09-02 — baseline (gpt-4o-mini judge, no reranker, top_k=10)

Config: `search.use_reranker=False`, `search.default_top_k=10`,
`search.vector_weight=1.0`, `search.bm25_weight=1.0` (repo defaults, unchanged).

| metric | value |
|---|---|
| hit_rate@10 | 1.000 (20/20) |
| mrr | 0.842 |
| context_precision | 0.889 |
| context_recall | 0.950 |

Per-question:

| id | difficulty | hit | mrr | ctx_precision | ctx_recall |
|---|---|---|---|---|---|
| q01 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q02 | easy | 1.0 | 0.500 | 0.583 | 1.000 |
| q03 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q04 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q05 | easy | 1.0 | 1.000 | 0.833 | 1.000 |
| q06 | easy | 1.0 | 1.000 | 0.917 | 1.000 |
| q07 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q08 | easy | 1.0 | 0.500 | 1.000 | 1.000 |
| q09 | easy | 1.0 | 0.500 | 1.000 | 1.000 |
| q10 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q11 | easy | 1.0 | 0.500 | 1.000 | 1.000 |
| q12 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q13 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q14 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q15 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q16 | easy | 1.0 | 0.333 | 0.583 | 1.000 |
| q17 | multi-hop | 1.0 | 1.000 | 1.000 | 1.000 |
| q18 | multi-hop | 1.0 | 0.500 | 0.867 | 1.000 |
| q19 | multi-hop | 1.0 | 1.000 | **0.000** | **0.000** |
| q20 | easy | 1.0 | 1.000 | 1.000 | 1.000 |
| q21 | negative | – | – | – | – |
| q22 | negative | – | – | – | – |
| q23 | negative | – | – | – | – |

**Finding — q19 (metric disagreement):** *"What state-of-the-art BLEU score
did the Transformer achieve on English-to-German translation, and what beam
size was used to obtain it during inference?"* (ground truth spans chunk 37 —
BLEU 28.4 — and chunk 38 — beam size 4). `hybrid_search`'s top-5 only
contained chunk 37; chunk 38 (beam size) never made it into the candidate
pool. `hit_rate` still scored this 1.0 because it only requires matching *any*
ground-truth chunk — it can't tell a fully-answered question from a
half-answered one. Ragas scored `context_precision`/`context_recall` at 0.0
for the same question, correctly reflecting that the retrieved context alone
can't support the full reference answer. This is the reason both metric
families are worth running together rather than picking one: a
purely-deterministic eval would have reported this question as a clean pass.

Not yet investigated: whether 0.0 (rather than a partial score reflecting
"BLEU claim supported, beam-size claim not") is Ragas correctly scoring a
compound claim as a single unit, or a judge-scoring artifact — worth a closer
look before leaning on `context_precision`/`context_recall` as a trusted
per-question signal rather than just an aggregate directional one.

**Negative controls:** for all three (implementation language/framework,
SQuAD accuracy, carbon footprint — none addressed by the paper),
`hybrid_search` returned topically-adjacent chunks (e.g. training
hardware/schedule for the carbon-footprint question) rather than anything
that reads as a confident, specific, wrong answer. No false-positive-looking
retrieval observed in this run.

## Known gaps in this eval (by design, see conversation history)

- Table/image chunks are excluded from the golden set entirely — no
  ground truth touches a `has_table`/`has_image` chunk yet.
- Only `hybrid_search` is covered. `similarity_search` (vector-only) and
  `generate_document_embedding` (ingestion correctness) have no eval yet.
- No comparison run with `use_reranker=True` or different
  `vector_weight`/`bm25_weight` — this baseline is the repo's current
  defaults only.
- 20 scored questions (+3 negative) is enough to sanity-check the harness,
  not enough to treat any single-point score change as statistically
  meaningful — treat deltas as directional until the set grows.
