# Document Upload Tool — Tasks
**Last updated:** 2026-07-20 · **Model:** claude-sonnet-5

## Dependency Graph

```
T1 config: DocumentConfig ─┐
T2 chunking: recursive_chunk ─┤
T3 parsing: extract_text (+ pypdf dep) ─┼──► T5 tool: upload_document (+ README)
T4 validation: new validators ─┘
```

T1–T4 have no dependencies on each other and can be implemented/reviewed independently (own stacked branch each, cut from the base branch). T5 depends on all four and is cut after they land.

**Extension (chunk_strategy, 2026-07-20):**

```
(T5, already landed) ──► T6: chunk_strategy support (config + validation + structural_chunk + tool wiring)
```

Done as a single task (combined by request) rather than split by layer — depends only on the already-landed T5.

## Tasks

### T1 — DocumentConfig

**Track:** config
**Depends on:** none
**Files (≤10):**
- `src/mcp_vectordb/config/config.py`
- `.env.example`
- `tests/test_config.py` (new or extended)

**Description:** Add a `DocumentConfig` pydantic model (`chunk_size: int = 500`, `chunk_overlap: int = 50`, `max_file_size_mb: float = 20.0`), wire it into `Settings` and `Settings.from_env()` reading `DOCUMENT_CHUNK_SIZE`, `DOCUMENT_CHUNK_OVERLAP`, `DOCUMENT_MAX_FILE_SIZE_MB`. Add the new env vars (commented, with defaults) to `.env.example`.

**TDD:**
- RED: test asserting `Settings.from_env()` produces a `document` config with defaults when env vars are unset, and picks up overrides when set (use `monkeypatch.setenv`).
- GREEN: implement `DocumentConfig` + wiring.
- REFACTOR: none expected — small, isolated change.

**Acceptance criteria:**
- `get_settings().document.chunk_size == 500` by default.
- Env var overrides are respected.
- Existing config tests still pass.

---

### T2 — Recursive chunking

**Track:** chunking
**Depends on:** none
**Files (≤10):**
- `src/mcp_vectordb/core/chunking.py` (new)
- `tests/test_chunking.py` (new)

**Description:** Implement `recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]`. Splits on `"\n\n"`, then `"\n"`, then `". "`, then falls back to a hard character window with overlap for any piece still longer than `chunk_size`. Must never return empty-string chunks, must never infinite-loop on pathological input (e.g. text with no separators), and `chunk_overlap < chunk_size` should be enforced (raise `ValueError` otherwise — validated again at the tool layer via T4, but the function itself should be safe standalone).

**TDD:**
- RED: tests for — short text (single chunk), text longer than chunk_size with paragraph breaks (chunks respect paragraph boundaries where possible), text with no separators at all (falls back to hard window + overlap), `chunk_overlap >= chunk_size` raises, empty string input raises/returns `[]` (pick one behavior and assert it), overlap actually reproduces overlapping content across adjacent chunks.
- GREEN: implement the splitter.
- REFACTOR: extract the separator cascade into a small internal helper if the function grows unwieldy.

**Acceptance criteria:**
- All tests above pass.
- No dependency added — pure stdlib implementation.

---

### T3 — Text extraction (PDF/TXT/MD)

**Track:** parsing
**Depends on:** none
**Files (≤10):**
- `src/mcp_vectordb/core/parsers.py` (new)
- `requirements.txt` (add `pypdf`)
- `tests/fixtures/sample.pdf`, `tests/fixtures/sample.txt`, `tests/fixtures/sample.md` (new, small test fixtures)
- `tests/test_parsers.py` (new)

**Description:** Implement `extract_text(file_path: str) -> List[Tuple[Optional[int], str]]` returning `(page_number, text)` pairs — `page_number` starts at 1 for PDFs, is `None` for `.txt`/`.md` (single entry). Dispatch by lowercased file extension; raise a clear error (`ValueError`, caught upstream by the tool as `ValidationError`) for unsupported extensions. Wrap `pypdf` parse failures in a distinct exception the tool layer can map to `RuntimeError`.

**TDD:**
- RED: tests — multi-page PDF fixture returns one tuple per page with correct page numbers and non-empty text; `.txt`/`.md` fixtures return a single `(None, text)` tuple with the exact file content; unsupported extension (e.g. `.docx`) raises; a corrupt/truncated PDF fixture raises a parse-failure exception (not a generic crash).
- GREEN: implement `extract_text` and the per-format helpers.
- REFACTOR: none expected.

**Acceptance criteria:**
- All tests above pass.
- `pypdf` added to `requirements.txt` with a version floor (`pypdf>=4.0.0`).

---

### T4 — Validators

**Track:** validation
**Depends on:** none
**Files (≤10):**
- `src/mcp_vectordb/utils/validation.py`
- `tests/test_validation.py` (extended)

**Description:** Add `validate_file_path(path: str, allowed_extensions: set, max_file_size_mb: float) -> str` (checks existence, readability, extension membership, size ≤ limit — all raising `ValidationError` with a clear message identifying which check failed). Add `validate_chunk_size(chunk_size: int) -> int` and `validate_chunk_overlap(chunk_overlap: int, chunk_size: int) -> int` (overlap must be `>= 0` and `< chunk_size`). Extend the reserved-keys set used by `validate_metadata` (or add a sibling `validate_document_metadata` if reuse isn't clean) to also reject: `document_id`, `source_filename`, `file_type`, `page_number`, `chunk_index`, `total_chunks`, `uploaded_at`.

**TDD:**
- RED: tests for each validator's happy path and every failure mode (missing file, wrong extension, oversized file, negative/zero chunk_size, overlap ≥ chunk_size, reserved metadata key rejected).
- GREEN: implement validators.
- REFACTOR: none expected.

**Acceptance criteria:**
- All tests above pass.
- Existing `validate_metadata`/`validate_text` tests still pass unchanged (no regressions from the reserved-key extension).

---

### T5 — `upload_document` MCP tool

**Track:** tool
**Depends on:** T1, T2, T3, T4
**Files (≤10):**
- `src/mcp_vectordb/tools/document.py` (new)
- `src/mcp_vectordb/tools/__init__.py` (register the new tool module)
- `README.md` (document the new tool, mirroring the `store_text`/`similarity_search` sections)
- `tests/test_upload_document.py` (new)

**Description:** Implement `@mcp.tool() async def upload_document(file_path, collection="documents", metadata=None, document_id=None, chunk_size=None, chunk_overlap=None, ctx=None) -> str` following the design's 10-step flow: validate inputs (T4 validators, using `DocumentConfig` defaults from T1 when `chunk_size`/`chunk_overlap` are `None`) → `ctx.info` start → `extract_text` (T3) per page → `recursive_chunk` (T2) per page with `ctx.debug` progress → build one `Document` per chunk with the full metadata set + merged user metadata → auto-create collection if missing (reusing the `store_text` pattern) → batch-embed via `embedding_service.generate_embeddings` → `vector_db.store_documents` → return a summary string → three-tier exception handling (`ValidationError`→`ValueError`, `VectorDBError`→`RuntimeError`, else→`RuntimeError`), each path logging via `ctx.error` when `ctx` is present, matching `store_text`/`similarity_search` exactly.

**TDD:**
- RED: tests (mocking `get_vector_db`/`get_embedding_service` the way existing tool tests do, if such mocks exist — otherwise establish the pattern here) — successful multi-page PDF upload produces N documents with correct chunk/page metadata and a summary mentioning `document_id`/`total_chunks`; successful `.txt` upload omits `page_number`; unsupported extension raises `ValueError`; oversized file raises `ValueError`; reserved metadata key in caller-supplied `metadata` raises `ValueError`; embedding/storage failure surfaces as `RuntimeError`.
- GREEN: implement `upload_document`.
- REFACTOR: extract any duplicated collection-auto-create logic shared with `store_text` into a small shared helper only if it's a clean, obvious extraction — otherwise leave the small duplication (matches the "don't force abstraction" guidance).

**Acceptance criteria:**
- All tests above pass; existing `store_text`/`similarity_search` tests unaffected.
- README's tool list documents `upload_document`'s parameters, matching the style of the existing two tool sections.
- Manual smoke check: running the server and calling `upload_document` against a real small PDF followed by `similarity_search` returns chunks with the expected metadata (documented as a manual step in the task's PR description, since no browser/UI is involved).

---

### T6 — `chunk_strategy` support (config + validation + `structural_chunk` + tool wiring)

**Track:** tool (combined, by request, rather than split by layer)
**Depends on:** T5 (landed)
**Files (≤10):**
- `src/mcp_vectordb/config/config.py`
- `.env.example`
- `src/mcp_vectordb/utils/validation.py`
- `src/mcp_vectordb/core/chunking.py`
- `src/mcp_vectordb/tools/document.py`
- `README.md` (document the new `chunk_strategy` parameter)
- `tests/test_config.py` (extended)
- `tests/test_validation.py` (extended)
- `tests/test_chunking.py` (extended)
- `tests/test_upload_document.py` (extended)

**Description:**
1. **Config:** add `chunk_strategy: str = "recursive"` to `DocumentConfig`, wired into `Settings.from_env()` reading `DOCUMENT_CHUNK_STRATEGY` (commented entry added to `.env.example`, same pattern as `DOCUMENT_CHUNK_SIZE`).
2. **Validation:** add `validate_chunk_strategy(chunk_strategy: str) -> str` to `utils/validation.py`, raising `ValidationError` (message naming the invalid value and the allowed set `{"recursive", "structural"}`) for anything outside that set.
3. **Chunking:** implement `structural_chunk(pages: List[Tuple[Optional[int], str]], chunk_size: int, chunk_overlap: int) -> List[Tuple[Optional[int], str]]` in `core/chunking.py`, alongside `recursive_chunk`. Input/output shape matches `extract_text`'s `(page_number, text)` list so it slots into `upload_document` without changing `parsers.py`.
   - If `len(pages) > 1` (PDF path — `extract_text` already gives one entry per page): pass each page through unchanged, one output chunk per page (page's full text, no further splitting even if it exceeds `chunk_size`).
   - If `len(pages) == 1` and `page_number is None` (`.md`/`.txt` path): split the single page's text on Markdown heading lines (regex `^#{1,6}\s.*$`, multiline) into sections — each section is the heading line plus everything up to the next heading or EOF, emitted as one `(None, section_text)` chunk each, whole (no size-based re-splitting, per the confirmed trade-off). Zero heading matches (all `.txt`, plus any `.md` without headings) produces exactly one section covering the whole text; for that no-heading case, delegate to `recursive_chunk(text, chunk_size, chunk_overlap)` and wrap each resulting piece as `(None, piece)` — so `.txt` output is unchanged from today's `recursive_chunk`-only behavior.
   - `chunk_overlap` is accepted for signature symmetry with `recursive_chunk` and used only in the no-heading fallback branch; it has no effect when structural splitting actually occurs (no overlap between sections/pages).
4. **Tool wiring:** add `chunk_strategy: Optional[str] = None` parameter to `upload_document`, resolved against `settings.document.chunk_strategy` when `None`, then validated via `validate_chunk_strategy` alongside the existing `chunk_size`/`chunk_overlap` validation, before any parsing (fail-fast, matching the existing validation-order convention). Replace the current per-page `recursive_chunk` loop with a strategy branch:
   - `"recursive"`: unchanged — call `recursive_chunk(page_text, ...)` per page, as today.
   - `"structural"`: call `structural_chunk(pages, chunk_size, chunk_overlap)` **once** for the whole document (not per page) and use that directly in place of the current per-page accumulation loop.
   Everything downstream (metadata assembly, embedding, storage, summary string, exception handling) is unchanged, since both branches converge back to the same `List[Tuple[Optional[int], str]]` shape the rest of the function already consumes.

**TDD:**
- RED: config — `Settings.from_env()` defaults to `chunk_strategy == "recursive"` when unset, picks up `DOCUMENT_CHUNK_STRATEGY` override via `monkeypatch.setenv`. Validation — `validate_chunk_strategy("recursive")`/`("structural")` pass through unchanged, `validate_chunk_strategy("bogus")` raises `ValidationError`. Chunking — markdown text with 3 `##` sections produces 3 chunks each starting with its heading line; markdown with zero headings falls back to `recursive_chunk`'s exact output (assert equality); multi-page input passes each page through as one chunk each with page numbers preserved, even for a page longer than `chunk_size`; single markdown section longer than `chunk_size` is kept whole (not split); empty input handled without crashing. Tool — `chunk_strategy="structural"` on a markdown fixture with headings produces chunks aligned to sections; on a multi-page PDF fixture produces one chunk per page; `chunk_strategy` omitted uses the `"recursive"` default and all existing T5 tests still pass unmodified; invalid `chunk_strategy` raises `ValueError` before any file parsing (assert `extract_text`/embedding mocks never called); `chunk_strategy="structural"` on a `.txt` fixture matches `"recursive"` output exactly.
- GREEN: implement config field, validator, `structural_chunk`, and the tool parameter/branch, in that order (each layer's tests pass before moving to the next).
- REFACTOR: extract the heading-split regex/loop into a small internal helper only if it collides in complexity with `_split_recursive`'s existing helpers; don't introduce a strategy-object abstraction for two call sites (per design decision 4, option A) — a dispatch branch is the minimal change.

**Acceptance criteria:**
- All tests above pass; all existing T1-T5 tests (`test_config.py`, `test_validation.py`, `test_chunking.py`'s `recursive_chunk` cases, `test_upload_document.py`) pass unmodified — default `chunk_strategy` behavior is byte-for-byte the same as before this task.
- README documents `chunk_strategy` with both allowed values and the default.
