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
