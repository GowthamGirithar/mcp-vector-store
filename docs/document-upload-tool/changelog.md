# Document Upload Tool — Changelog

## [2026-07-20] — Spec created
Model: claude-sonnet-5
Trigger: initial feature request — add document upload/chunk/embed/store tool
Summary: Confirmed spec for `upload_document` tool: server-local file path input, PDF/.txt/.md support, recursive/semantic chunking, synchronous processing with ctx progress, fail-fast validation, full metadata set (document_id, source_filename, file_type, page_number, chunk_index, total_chunks, uploaded_at), chunk config via new DocumentConfig.

## [2026-07-20] — Design created
Model: claude-sonnet-5
Trigger: spec confirmed, moved to design phase
Summary: Confirmed design: plain functions in new core/parsers.py + core/chunking.py (no new ABC), pypdf for PDF extraction, custom recursive character splitter (no LangChain dependency), new DocumentConfig, new tools/document.py::upload_document following the store_text/similarity_search error-handling and ctx-logging conventions.

## [2026-07-20] — Tasks created
Model: claude-sonnet-5
Trigger: design confirmed, moved to task breakdown
Summary: Confirmed 5-task breakdown — T1 DocumentConfig, T2 recursive_chunk, T3 extract_text (+pypdf), T4 validators, T5 upload_document tool wiring (depends on T1-T4). T1-T4 are independent and can proceed in parallel stacked branches; T5 is cut after they land.

## [2026-07-20] — T1 implemented: DocumentConfig
Model: claude-sonnet-5
Branch: document-upload-tool/t1-document-config (pushed, no PR — see note below)
Summary: Added `DocumentConfig` pydantic model (chunk_size=500, chunk_overlap=50, max_file_size_mb=20.0) to `src/mcp_vectordb/config/config.py`, wired into `Settings` and `Settings.from_env()`, reading `DOCUMENT_CHUNK_SIZE`, `DOCUMENT_CHUNK_OVERLAP`, `DOCUMENT_MAX_FILE_SIZE_MB` env vars with the same defaults as fallback. Added the corresponding commented entries to `.env.example`.
Test-setup decisions (relevant for T2-T5): this repo had no `tests/` directory and no pytest config at all, so this task created both from scratch — root `pytest.ini` (`pythonpath = src`, `testpaths = tests`, so tests import as `from mcp_vectordb...`), `tests/conftest.py` with an autouse `reset_settings_cache` fixture (resets the `get_settings()` singleton between tests), and a local gitignored `.venv`.
Note: `gh pr create` failed — the authenticated `gh` account has no access to this repo (different account than the repo owner). Branch is implemented, tested, committed, and pushed; PR creation deferred (decided to skip PR creation for the whole feature for now and open PRs manually later).

## [2026-07-20] — T2 implemented: Recursive chunking
Model: claude-sonnet-5
Branch: document-upload-tool/t2-recursive-chunking (pushed, no PR)
Summary: Implemented `recursive_chunk(text, chunk_size, chunk_overlap)` in `src/mcp_vectordb/core/chunking.py`. Splits on `\n\n` (paragraphs), then `\n` (lines), then `". "` (sentences), then falls back to a hard character window with overlap for any piece still too long. Empty string input returns `[]`. Raises `ValueError` for `chunk_size <= 0`, `chunk_overlap < 0`, or `chunk_overlap >= chunk_size`. Hard-window fallback advances by `chunk_size - chunk_overlap` characters per step (always ≥1 given validated inputs), guaranteeing termination even on pathological no-separator input. Overlap correctness verified by asserting the tail of chunk[i] appears at the start of chunk[i+1]. Separator-split pieces are greedily re-merged up to chunk_size, so a chunk may contain multiple small paragraphs (not strictly one-paragraph-per-chunk) — relevant for T5's assumptions about chunk boundaries. No new dependency (pure stdlib).
Test setup: cut from `main` (T1's `tests/`/`pytest.ini` not yet merged there), so recreated the same `pytest.ini` convention independently. No `conftest.py` needed (pure function, no fixtures).

## [2026-07-20] — T3 implemented: Text extraction (PDF/TXT/MD)
Model: claude-sonnet-5
Branch: document-upload-tool/t3-text-extraction (pushed, no PR)
Summary: Implemented `extract_text(file_path) -> List[Tuple[Optional[int], str]]` in `src/mcp_vectordb/core/parsers.py`. Dispatches on lowercased extension: `.pdf` returns one `(page_number, text)` tuple per page (1-indexed) via `pypdf.PdfReader`; `.txt`/`.md` return a single `(None, full_text)` tuple. Unsupported extensions raise `UnsupportedFileTypeError` (a `ValueError` subclass) before the file is opened. Corrupt/truncated PDFs raise `DocumentParseError` (wraps `pypdf.errors.PdfReadError`/`PdfStreamError`) — **T5 should catch `DocumentParseError` and map it to `RuntimeError`**, and catch `UnsupportedFileTypeError` separately since it's intentionally a distinct `ValueError` subtype. Added `pypdf>=4.0.0` to `requirements.txt`. Test fixtures added under `tests/fixtures/`: `sample.txt`, `sample.md`, `sample.docx` (empty, dispatch-failure test only), `sample.pdf` (valid 3-page PDF with real extractable text, generated via `reportlab`, dev-only, not added to requirements-dev.txt), `corrupt.pdf` (truncated copy of `sample.pdf`).
Test setup: cut from `main` (same as T1/T2, no `tests/`/`pytest.ini` there yet), recreated the same convention independently.

## [2026-07-20] — T4 implemented: Validators
Model: claude-sonnet-5
Branch: document-upload-tool/t4-validators (pushed, no PR)
Summary: Added `validate_file_path(path, allowed_extensions, max_file_size_mb)`, `validate_chunk_size(chunk_size)`, and `validate_chunk_overlap(chunk_overlap, chunk_size)` to `src/mcp_vectordb/utils/validation.py`, all raising `ValidationError` with a specific message per failure mode. Extended `validate_metadata`'s existing `reserved_keys` set directly (no sibling function) to also reject `document_id`, `source_filename`, `file_type`, `page_number`, `chunk_index`, `total_chunks`, `uploaded_at` — confirmed this doesn't affect `store_text` in `tools/storage.py`, which never uses those key names.
Test setup: `.venv` and pytest were already present from prior task setup in the working tree; only needed to add `pytest.ini` (not yet tracked on `main`) and `tests/test_validation.py`.

## [2026-07-20] — Orchestration note: docs/ ownership
Trigger: T1, T2, and T3 each independently recreated/diverged `docs/document-upload-tool/changelog.md` on their own branches, because the folder is untracked on `main` and disappears from the working tree whenever a branch that *did* commit it is checked out away from (git only keeps files present in the target branch's tree). This produced three divergent copies of this changelog across branches.
Resolution: the orchestrator (not sub-agents) now owns `docs/document-upload-tool/` exclusively. This file has been manually reconciled into one canonical history. T4/T5 sub-agents are instructed not to touch `docs/` at all — the orchestrator appends their changelog entries after each task completes, based on the sub-agent's returned summary.

## [2026-07-20] — T2 revised: word-level fallback replaces sentence-level
Model: claude-sonnet-5
Branch: document-upload-tool/t5-upload-document-tool
Trigger: explicit request to change the separator hierarchy to paragraphs → lines → words, dropping sentence-level (`". "`) splitting.
Summary: In `src/mcp_vectordb/core/chunking.py`, replaced the `". "` sentence-separator tier with a `" "` word-separator tier (`_split_by_sentence` renamed to `_split_by_word`, `_SENTENCE_SEP` renamed to `_WORD_SEP`); hard-window fallback is now only reached when a single word exceeds `chunk_size`. Updated module/function docstrings accordingly. `tests/test_chunking.py` updated via Red-Green-Refactor: added `word_split_when_line_too_long` (replacing the old sentence-split case), and changed the hard-window/overlap/pathological-input fixtures from space-containing phrases to single long invented words (e.g. `"pneumonoultramicroscopicsilicovolcanoconiosis"`) since ordinary space-containing text no longer reaches the hard-window tier. All 14 cases pass.

## [2026-07-20] — Spec + Design updated: chunk_strategy extension
Model: claude-sonnet-5
Trigger: new requirement — upload_document should support a selectable chunk_strategy, with a document-based/structural strategy alongside the existing recursive one
Summary: Extended spec.md with a "structural" chunking option: `.md` splits on Markdown headings (one chunk per section), `.pdf` treats each existing per-page extraction as one chunk, `.txt` (no structural signal) falls back to recursive_chunk output unchanged. Oversized structural sections are kept whole (semantic coherence over the chunk_size contract — confirmed trade-off, not further split). New `chunk_strategy: Optional[str]` tool param defaults to `"recursive"` via a new `DocumentConfig.chunk_strategy` / `DOCUMENT_CHUNK_STRATEGY` default, fully backward compatible. Extended design.md with the chosen approach: a new `structural_chunk(pages, chunk_size, chunk_overlap)` function in `core/chunking.py` (operates on `extract_text`'s already-parsed `(page_number, text)` list, not raw files) plus a small strategy-dispatch dict in `tools/document.py`, and a new `validate_chunk_strategy` validator — rejected a `ChunkStrategy` ABC as premature for two strategies. Manifest reset to draft for Spec/Design/Tasks pending user confirmation.

## [2026-07-20] — Spec/Design confirmed; Tasks combined into single T6
Model: claude-sonnet-5
Trigger: user confirmed the chunk_strategy spec + design; user then requested T6-T8 be combined into one task (matches internal convention of one task per unit of work here) rather than split by layer.
Summary: Spec and Design marked confirmed. Task breakdown collapsed from three tasks (T6 config/validation, T7 structural_chunk, T8 tool wiring) into a single T6 covering all four pieces (DocumentConfig.chunk_strategy + DOCUMENT_CHUNK_STRATEGY, validate_chunk_strategy, structural_chunk in core/chunking.py, and chunk_strategy wiring in tools/document.py + README), depending only on the already-landed T5. Tasks confirmed; proceeding to local implementation (Setup step, then a sub-agent implements T6 end-to-end on its own stacked branch).
