# Generate Document Embedding — Changelog

## 2026-07-25 — Spec created
Model: claude-sonnet-5
Trigger: initial feature request — multimodal document processing tool
Summary: Clarified scope down to extraction-only (text/table/image separation via `unstructured`), no embedding/storage in this phase. Tool named `generate_document_embedding` (forward-looking name; embedding is a future phase). Spec confirmed by user.

## 2026-07-25 — Design created
Model: claude-sonnet-5
Trigger: design phase following confirmed spec
Summary: Chose Approach #1 (new core module + new tool file, mirroring core/parsers.py + tools/storage.py split). User requested file naming `tools/document_embedding.py` instead of `tools/extraction.py`; core module renamed to `core/document_embedding.py` for consistency. Design confirmed by user.

## 2026-07-25 — Tasks created
Model: claude-sonnet-5
Trigger: task breakdown following confirmed design; user requested a single collapsed task instead of the initially proposed 4-task split
Summary: Collapsed dependency, fixtures, core module, and tool into one TDD task (T1) covering the full RED-GREEN-REFACTOR cycle. Tasks confirmed by user.

## 2026-07-25 — Spec/design revised during implementation
Model: claude-sonnet-5
Trigger: empirical testing against generated fixtures during T1 implementation surfaced two real constraints not known at design time
Summary: (1) unstructured 0.24.1 requires `tesseract` (installed via Homebrew) plus `strategy="hi_res"` for reliable Table/Image detection on PDFs — `poppler` alone (already present) was insufficient. (2) unstructured has no built-in DOCX/PPTX image extraction (pluggable extension point, no shipped implementation) — user chose to accept this as a documented known limitation rather than write custom picture-partitioner plugins. spec.md and design.md updated accordingly; both remain confirmed.
