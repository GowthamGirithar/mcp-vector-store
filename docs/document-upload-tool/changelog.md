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
Branch: document-upload-tool/t1-document-config
Summary: Added `DocumentConfig` pydantic model (chunk_size=500, chunk_overlap=50, max_file_size_mb=20.0) to `src/mcp_vectordb/config/config.py`, wired into `Settings` and `Settings.from_env()`, reading `DOCUMENT_CHUNK_SIZE`, `DOCUMENT_CHUNK_OVERLAP`, `DOCUMENT_MAX_FILE_SIZE_MB` env vars with the same defaults as fallback. Added the corresponding commented entries to `.env.example`.
Test-setup decisions (relevant for T2-T5): this repo had no `tests/` directory and no pytest config at all, so this task created both from scratch:
- `tests/` lives at repo root (sibling to `src/`), with `tests/conftest.py` and `tests/test_config.py`.
- Added root-level `pytest.ini` with `pythonpath = src` and `testpaths = tests`. This puts `src/` on `sys.path` for test collection, so tests import as `from mcp_vectordb.config.config import ...` (no `src.` prefix needed), even though `main.py` itself imports via `from src.mcp_vectordb...`. Follow this same import style (`from mcp_vectordb...`, not `from src.mcp_vectordb...`) in new test files.
- `get_settings()` caches a module-level singleton in `mcp_vectordb.config.config._settings`. Added an autouse fixture in `tests/conftest.py` (`reset_settings_cache`) that resets `config_module._settings = None` before and after every test, so `monkeypatch.setenv(...)` changes are actually picked up by `get_settings()` in later tests. This fixture is autouse and repo-wide, so T2-T5 do not need to redeclare it — just reuse `tests/conftest.py`.
- No virtualenv existed in the repo; created a local `.venv` (already gitignored) and installed `requirements.txt` + `requirements-dev.txt` into it to run `pytest`. `pytest-asyncio` was not needed for this task (no async code under test).
- The repo's `docs/document-upload-tool/` directory (spec.md, design.md, manifest.md, tasks.md, tasks.json, changelog.md) existed only as untracked files on `main` before this branch — committed them here so they're available on `main`/this stacked branch set going forward.
