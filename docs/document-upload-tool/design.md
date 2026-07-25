# Document Upload Tool — Design
**Last updated:** 2026-07-20 · **Model:** claude-sonnet-5

## Approaches Considered

### 1. Module structure: plain functions vs. extending `DocumentProcessor` ABC

**A. Plain functions in new modules (chosen)**
- `core/parsers.py::extract_text(path) -> List[PageText]` dispatches by file extension (`.pdf` → `pypdf`, `.txt`/`.md` → direct read).
- `core/chunking.py::recursive_chunk(text, chunk_size, chunk_overlap) -> List[str]`.
- Pros: no interface reshaping, no abstraction for a single implementation, easy to unit test each function in isolation, matches existing lightweight style of `utils/validation.py`.
- Cons: if a second processor type appears later (e.g. streaming ingestion), the boundary would need to be introduced then.

**B. Extend `DocumentProcessor` ABC**
- Reshape `process_text`/`process_texts` to return chunked, paged output; implement `FileDocumentProcessor`.
- Pros: pluggable if more processor implementations show up.
- Cons: forces a currently-unused interface to bend around one new caller; premature — YAGNI given there's exactly one processing path today.

**Chosen: A.** Simpler, matches codebase conventions, no speculative extensibility.

### 2. PDF text extraction library

**A. `pypdf` (chosen)**
- Pure-Python (no native/C build step), actively maintained fork of the old PyPDF2, already a common, low-risk dependency for this scope. `PdfReader(path).pages[i].extract_text()` gives per-page text directly, which lines up with the `page_number` metadata requirement.
- Cons: extraction quality is workable but not as good as `pymupdf` on complex layouts (multi-column, tables).

**B. `pymupdf` (fitz)**
- Pros: generally better text-extraction fidelity.
- Cons: heavier dependency (bundles MuPDF, a C library), AGPL/commercial licensing considerations, more than this scope needs.

**Chosen: A** (`pypdf`) — extraction quality is sufficient for the goal (semantic search over chunks, not layout-perfect reproduction), and it keeps the dependency footprint and licensing simple, consistent with `requirements.txt`'s current pure-Python-first set.

### 3. Chunking implementation

**A. Custom recursive splitter (chosen)**
- Try splitting on `"\n\n"` (paragraphs), then `"\n"` (lines), then `" "` (words), then hard character-window fallback with overlap — same core idea as LangChain's `RecursiveCharacterTextSplitter`, implemented directly (~40 lines) with no new dependency.
- Pros: zero new dependency, full control, easy to unit test.
- Cons: reimplements a well-known algorithm instead of reusing a battle-tested library.

**B. `langchain-text-splitters` package**
- Pros: battle-tested, more separator/language-aware splitting options.
- Cons: new dependency (even the standalone package pulls in its own version-compat surface) for a single function's worth of value; codebase currently has zero LangChain dependencies.

**Chosen: A** — the algorithm is simple enough to own directly, and it avoids adding a new dependency family for one utility function.

### 4. Pluggable chunk strategy (Extension, 2026-07-20)

**A. Strategy dispatch function + new `structural_chunk` in `core/chunking.py` (chosen)**
- Add `structural_chunk(pages, chunk_size, chunk_overlap) -> List[Tuple[Optional[int], str]]` next to `recursive_chunk` in `core/chunking.py`. It operates on the already-extracted `(page_number, text)` list (not raw file bytes), so it stays parser-agnostic:
  - `.pdf` (page_number is not None): each page's text becomes exactly one output chunk, unchanged — `extract_text` already gives per-page granularity.
  - `.md`/`.txt` (page_number is None, single input page): split the page text on Markdown heading lines (regex `^#{1,6}\s`) into sections; each section (heading line + its body, up to the next heading or EOF) becomes one chunk. A file with zero headings (including all `.txt` files) yields one section — the whole text — which is intentionally >chunk_size-agnostic, then handed to `recursive_chunk` as the fallback (see open question below) so `.txt` behavior is unchanged from today.
  - Oversized sections are **not** further split (confirmed trade-off) — semantic coherence over the `chunk_size` contract.
- `tools/document.py` gains a small dispatch: `_CHUNKERS = {"recursive": ..., "structural": ...}`, chosen by validated `chunk_strategy`, called once per document (structural needs the whole `pages` list, unlike `recursive_chunk` which is called per-page today) — this changes the per-page loop in `upload_document` to branch: `recursive` keeps the current per-page `recursive_chunk` call; `structural` calls a single `structural_chunk(pages, ...)` covering all pages at once.
- Pros: keeps `recursive_chunk`'s existing contract and all T2 tests untouched; new function is independently unit-testable; no change to `extract_text`/`parsers.py`.
- Cons: `upload_document`'s page-processing loop needs a strategy-shaped branch instead of one uniform call.

**B. `ChunkStrategy` ABC with `RecursiveChunkStrategy`/`StructuralChunkStrategy` classes**
- Pros: textbook Strategy pattern, easy to add a third strategy later purely by subclassing.
- Cons: two strategies is not enough to justify a class hierarchy yet (same YAGNI reasoning as approach 1's rejected option B); a plain dict-of-functions dispatch gives the same swap-ability with far less ceremony.

**Chosen: A** — small function + dispatch dict, consistent with the codebase's existing "plain functions" convention (see decision 1 above), no premature class hierarchy for two strategies.

**Chunk strategy validation:** `validate_chunk_strategy(chunk_strategy: str) -> str` added to `utils/validation.py` alongside `validate_chunk_size`/`validate_chunk_overlap`, raising `ValidationError` if not in `{"recursive", "structural"}`.

**Config:** `DocumentConfig.chunk_strategy: str = "recursive"`, loaded from `DOCUMENT_CHUNK_STRATEGY`, same override pattern as `chunk_size`/`chunk_overlap` (tool param wins, falls back to config default).

## Chosen Approach — Summary

```
src/mcp_vectordb/
├── config/config.py           # + DocumentConfig (chunk_size, chunk_overlap, max_file_size_mb)
├── core/
│   ├── parsers.py             # NEW: extract_text(path) -> List[PageText(page_number, text)]
│   └── chunking.py            # NEW: recursive_chunk(text, chunk_size, chunk_overlap) -> List[str]
├── utils/
│   ├── validation.py          # + validate_file_path, validate_chunk_size, validate_chunk_overlap
│   └── exceptions.py          # reuse existing ValidationError/VectorDBError
└── tools/
    └── document.py            # NEW: @mcp.tool() upload_document(...)
```

**Flow inside `upload_document`:**
1. Validate `file_path` (exists, extension supported, size ≤ `max_file_size_mb`), `collection` name, and `metadata` (reserved-key check) — fail fast, before any parsing.
2. `ctx.info` — starting.
3. `extract_text(file_path)` → list of `(page_number | None, text)` pages. `.txt`/`.md` yield a single page with `page_number=None`.
4. For each page, `recursive_chunk(text, chunk_size, chunk_overlap)` → list of chunk strings; `ctx.debug` progress per page.
5. Assign a single `document_id` (uuid4) for the whole upload; build one `Document` per chunk with `metadata = {document_id, source_filename, file_type, page_number, chunk_index, total_chunks, uploaded_at, **user_metadata}`.
6. Auto-create the collection if missing (same pattern as `store_text`, using `embedding_service.dimension`).
7. Batch-embed all chunk texts via `embedding_service.generate_embeddings(texts)` (already exists, uses the `CachedEmbeddingService` wrapper transparently).
8. `vector_db.store_documents(documents, collection)` (existing batch method — no adapter changes needed).
9. Return a summary string: `document_id`, `collection`, pages processed, `total_chunks`, filename — mirroring the response style of `store_text`.
10. Same three-tier exception handling as `store_text`/`similarity_search`: `ValidationError` → `ValueError`, `VectorDBError` → `RuntimeError`, anything else → `RuntimeError`, each logged via `ctx.error` when `ctx` is present.

**New dependency:** `pypdf` added to `requirements.txt`.

**Config additions (`config/config.py`):**
```python
class DocumentConfig(BaseModel):
    chunk_size: int = Field(default=500)        # chars
    chunk_overlap: int = Field(default=50)       # chars
    max_file_size_mb: float = Field(default=20.0)
```
Loaded from `DOCUMENT_CHUNK_SIZE`, `DOCUMENT_CHUNK_OVERLAP`, `DOCUMENT_MAX_FILE_SIZE_MB` env vars, with `chunk_size`/`chunk_overlap` overridable as optional `upload_document` tool parameters (falling back to config defaults, matching how `collection`/`top_k` defaults work in the existing tools).

**Reserved metadata keys** (cannot be overridden by caller-supplied `metadata`, extending the existing `validate_metadata` reserved-key set): `document_id`, `source_filename`, `file_type`, `page_number`, `chunk_index`, `total_chunks`, `uploaded_at`.

## Risks & Open Questions

- **[Extension] Structural mode can exceed `chunk_size` for large sections** — accepted trade-off per spec; downstream embedding-model token limits are not enforced here (same character-length reasoning as the base feature's own open question below). If a section is large enough to exceed the embedding model's context window, `generate_embeddings` would surface that failure — out of scope to guard against in this extension.
- **[Extension] `.txt` files see no behavioral change under `structural`** — since there's no structural signal to split on, `structural` degrades to `recursive_chunk` output for `.txt`. This is called out explicitly in the spec so it isn't mistaken for a bug.
- **`pypdf` extraction quality on scanned/complex-layout PDFs** — out of scope per spec non-goals (no OCR); text-only PDFs are the target.
- **Large PDFs (hundreds of pages) processed synchronously** — could make the tool call long-running; mitigated by `ctx` progress reporting so the client at least sees liveness, and by `max_file_size_mb` as a coarse guard. If this becomes a real problem later, the background-job approach (rejected in the spec interview) is the natural escalation path.
- **Chunk size in characters vs. tokens** — using character-based windows (like the rest of the spec implies) is simpler but not token-precise; acceptable given `store_text`'s own `validate_text` also uses a character-length limit (100,000).

## Lifecycle / Removal Plan

Not applicable — this is additive (new tool, new modules, new config), no existing behavior is removed or deprecated.
