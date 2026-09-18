# TurboQuant compression & shadow evaluation

Status: experimental, off by default (`COMPRESSION_ENABLED=false`, `COMPRESSION_SHADOW_ENABLED=false`).

## What this is

A post-hoc, model-agnostic compression layer applied to embeddings after
they're generated and before storage. It does not replace the embedding
model — it compresses whatever float vector the model produces, to shrink
stored vector size and cheapen approximate similarity scoring, while
keeping the exact-vector path untouched as the source of truth.

### On storage

When compression is active (`enabled` or `shadow_enabled`), each document's
full-precision embedding is additionally run through a fixed random
orthogonal rotation (deterministic per `(dimension, seed)`, decorrelates the
vector's dimensions) and then per-vector scalar quantization, packing each
rotated dimension into `bits` (1–8, default 8). This produces `uint8` codes
plus a per-vector `vmin`/`scale` needed to approximately reconstruct the
original vector. The codes are stored alongside the document's existing
metadata; the full-precision embedding is still stored and indexed as usual
for the real search path — compression adds a parallel, approximate copy,
it does not replace anything.

### On retrieval

Two independent paths read the stored codes:

- **Stage-1 compressed scan** — every stored `(doc_id, code)` pair for a
  collection is fetched, decoded, and scored against the raw query vector
  by decoding and cosine-scoring the reconstruction (no separate ANN
  structure understands the codes directly). This is a linear scan, not an
  index lookup — affordable only because the codes are small and the score
  is cheap. The top-N results feed into RRF fusion with BM25.
- **Shadow comparison** — when `shadow_enabled`, this compressed+BM25
  ranking (no exact rerank) is computed alongside every real `hybrid_search`
  call and compared to the real ranking. It never blocks or influences the
  real response: it runs fire-and-forget, and any exception is caught and
  logged, never raised.

## How it's wired today

- `CompressionConfig` (`config/config.py`): `enabled`, `shadow_enabled`,
  `bits`, `seed`, `candidate_pool_size`. `enabled` controls whether codes are
  generated/stored at write time; `shadow_enabled` controls whether the
  shadow comparison runs at query time. Either flag being on triggers code
  generation on write (`ChromaAdapter._compression_active`).
- **Storage (`adapters/chroma.py`)**: on `store_documents`, if compression is
  active, each embedding is encoded and the codes are stashed as **Chroma
  document metadata** — `_turboquant_codes` (base64 of the raw `uint8`
  bytes), `_turboquant_vmin`, `_turboquant_scale` — alongside the document's
  real metadata and its full-precision embedding (which Chroma still stores
  and indexes as usual for the real search path). These keys are stripped
  before metadata is returned to callers (`_strip_internal_metadata`).
- **Retrieval**: `get_all_quantized_vectors(collection, filters)` fetches
  every document's metadata for a collection, decodes the base64 codes back
  into `QuantizedVector`s, and hands them to `turboquant_scan.scan_top_n`.
  This is a full metadata scan per shadow query — no index, no filtering
  pushed to a vector-native structure.
- **Trigger point**: `tools/search.py::_hybrid_search_traced` fires the
  shadow run via `asyncio.create_task` (fire-and-forget) after computing the
  real result, only when `compression_config.shadow_enabled`.

## Observing the shadow run

The shadow run never returns anything to the caller — it's only visible
through logs (logger name `mcp_vectordb.search.shadow`, level `INFO`).

- **Skipped** (adapter has no code storage support, or no codes stored yet
  for the collection): a single `INFO` line naming the query/collection and
  the skip reason.
- **Completed**: an `INFO` line per `hybrid_search` call with
  `overlap_at_k` — the fraction of the real ranking's doc IDs also present
  in the shadow ranking — plus the full `real=` and `shadow=` ID lists for
  that call, so a specific query's divergence can be inspected directly.
- **Failed**: an `ERROR` line (via `logger.exception`) with the query,
  collection, and stack trace. This never affects the real response.

There's no aggregation across calls today (see item 7 below) — observing
trends over time means collecting these `overlap_at_k` values from logs
yourself (e.g. grep/parse and average over a query sample) rather than
reading a dashboard.

## Why "store as vector instead of encoded" is the next step

Codes currently live as opaque base64 strings in Chroma's metadata store —
a workaround, not a real compressed-vector storage path. That means:

- No ANN index (HNSW or otherwise) is ever built over the codes; every
  shadow query is an O(collection size) metadata fetch + linear scan.
- Codes ride inside the metadata JSON blob per document, which is
  Chroma's general-purpose key/value store, not a columnar/vector store —
  it doesn't get the compact fixed-width layout a real vector column would.
- There's no dedicated versioning/dimension check at the storage layer
  beyond what `_get_codec` does in-process; codes from different
  `(bits, seed)` configs would silently coexist if config changed without
  a re-encode.

The stated future direction is to store the quantized codes as an actual
vector type (e.g. a dedicated ANN-indexable column/field) instead of an
encoded metadata string, so stage-1 retrieval can use an index instead of a
linear scan.

## What to investigate before that migration

1. **Target vector store's support for sub-float / binary vector types.**
   Does Chroma (or whatever store is targeted) support indexing `uint8` or
   packed-bit vectors natively, with a distance metric that matches
   `TurboQuantCodec.score`'s approximate-cosine semantics? If not, which
   store does (e.g. pgvector's `halfvec`/binary quantization support,
   Qdrant/Milvus scalar or binary quantization, Elasticsearch/OpenSearch
   `byte`/`bit` vectors) — and what migration cost that implies away from
   the current Chroma-only adapter.
2. **Distance metric fidelity.** `codec.score` currently decodes codes back
   to floats and computes exact cosine similarity on the reconstruction. A
   native vector index typically scores directly in the compressed space
   (e.g. Hamming/asymmetric distance for binary codes, or a custom SIMD
   kernel) — confirm the target index's native metric preserves ranking
   quality close enough to today's decode-then-score approach, or whether
   the codec's encoding scheme (rotation + scalar quant) needs to change to
   match what the index can score efficiently.
3. **Bit-packing.** Codes are currently stored one `uint8` per dimension
   regardless of `bits` (a `bits=1` config still writes a full byte per
   dimension via `_encode_codes`). A real vector-column migration should
   also decide whether to pack sub-byte codes tightly (e.g. 8 dimensions
   per byte for `bits=1`), since that's most of the storage-size argument
   for quantization in the first place.
4. **Index build/maintenance cost.** HNSW-style index construction over
   quantized vectors has its own memory/CPU cost and needs to be kept in
   sync with writes/deletes — figure out whether the target store maintains
   this incrementally or requires periodic rebuilds, and how that interacts
   with this project's existing per-collection write path
   (`store_documents`, `delete_documents`).
5. **Re-indexing / migration path for existing data.** Documents written
   before compression was enabled (or under a different `bits`/`seed`) have
   no codes today (`get_all_quantized_vectors` simply skips them — see the
   docstring in `chroma.py`). A move to a vector-native column needs an
   explicit backfill job, not just "the next write includes it," and a
   decision on what happens to collections with mixed/no codes during the
   transition.
6. **Filter pushdown.** Metadata filters (`filters=...`) currently apply via
   Chroma's `where` clause on the same document row that holds the codes.
   If codes move to a separate vector-only structure, confirm filtered
   shadow/stage-1 queries can still be scoped without falling back to a
   full unfiltered scan plus post-filtering.
7. **Promotion criteria from shadow to real.** Before flipping
   `compression.enabled` on for real traffic (not just shadow), define the
   `overlap_at_k` (and any added recall/precision-at-k) threshold, over what
   query volume/time window, that constitutes "good enough" — this doc's
   shadow path only logs the metric today; there's no aggregation or
   dashboard yet.

## Config reference

| Env var | Field | Default | Notes |
|---|---|---|---|
| `COMPRESSION_ENABLED` | `compression.enabled` | `false` | Turns on code generation at write time. Enabling for an existing collection does not retroactively encode already-stored documents. |
| `COMPRESSION_SHADOW_ENABLED` | `compression.shadow_enabled` | `false` | Runs the shadow comparison per `hybrid_search` call; also triggers code generation at write time. |
| `COMPRESSION_BITS` | `compression.bits` | `8` | 1–8 bits per dimension. |
| `COMPRESSION_SEED` | `compression.seed` | `42` | Must stay fixed for a given collection's codes to remain decodable consistently. |
| `COMPRESSION_CANDIDATE_POOL_SIZE` | `compression.candidate_pool_size` | `200` | Stage-1 shortlist size fed into shadow RRF fusion. |
