# Generate Document Embedding — Tasks
**Last updated:** 2026-07-25 · **Model:** claude-sonnet-5

## Dependency Graph

Single task, no cross-task dependencies.

## Tasks

### T1 — Implement `generate_document_embedding` end-to-end

**Description:** Add the `unstructured` dependency, build the multimodal test fixtures,
implement the core extraction module, and implement the MCP tool that wraps it — all in
one task/commit sequence, per the confirmed design (Approach #1: `core/document_embedding.py`
+ `tools/document_embedding.py`).

**Files:**
- `requirements.txt` — add `unstructured[pdf,docx,pptx,md]`
- `requirements-dev.txt` — add `reportlab` (fixture generation only)
- `tests/fixtures/multimodal_sample.pdf`, `multimodal_sample.docx`, `multimodal_sample.pptx`,
  `multimodal_sample.md` — each with at least one text paragraph, one table, and (for
  pdf/docx/pptx) one embedded image
- `scripts/generate_multimodal_fixtures.py` — one-off generator for the above fixtures
  (not part of the test suite)
- `src/mcp_vectordb/core/document_embedding.py` — `MULTIMODAL_SUPPORTED_EXTENSIONS`,
  `ExtractedElement`, `MultimodalExtractionResult`, `DocumentEmbeddingParseError`,
  `extract_multimodal()`
- `tests/test_document_embedding_core.py`
- `src/mcp_vectordb/tools/document_embedding.py` — `generate_document_embedding` MCP tool
- `src/mcp_vectordb/tools/__init__.py` — register the new tool module
- `tests/test_document_embedding_tool.py`

**TDD approach:**
1. RED: write core-module tests against the fixtures (category counts/content shape per
   file type, unsupported-extension error, corrupt-file error, empty-document case) —
   failing since `core/document_embedding.py` doesn't exist yet.
2. GREEN: implement `core/document_embedding.py` to pass those tests.
3. RED: write tool-level tests (validation error paths, parse error path, happy-path
   summary string counts) against the real core function and fixtures — no mocking needed
   since there's no embedding/storage dependency to fake.
4. GREEN: implement `tools/document_embedding.py` and register it in `tools/__init__.py`.
5. REFACTOR: clean up, confirm full test suite passes.

**Acceptance criteria:** matches every GIVEN/WHEN/THEN scenario in `spec.md`.
