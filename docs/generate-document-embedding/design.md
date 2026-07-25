# Generate Document Embedding — Design
**Last updated:** 2026-07-25 · **Model:** claude-sonnet-5

## Approaches Considered

1. **New core module + new tool file (mirrors existing pattern)** — Add
   `core/document_embedding.py` with a pure function
   `extract_multimodal(file_path) -> MultimodalExtractionResult` (a small model holding
   `text_elements`, `table_elements`, `image_elements`, each an `ExtractedElement` with
   `category`, `element_type`, `page_number`, `content`). New `tools/document_embedding.py`
   registers `generate_document_embedding`, doing validation + calling the core function +
   logging + building the summary string.
   - *Pros:* Matches the existing `core/parsers.py` (parsing) vs `tools/storage.py`
     (MCP-facing wrapper) separation. Core function is unit-testable in isolation, same as
     `test_parsers.py` does today, without needing a FastMCP context. File names
     (`core/document_embedding.py` / `tools/document_embedding.py`) stay consistent with the
     tool's own name (`generate_document_embedding`), matching how `core/chunking.py` and
     `core/parsers.py` are named after what they do rather than a generic term.
   - *Cons:* Introduces new core module/models that may need reshaping once a later
     embedding phase builds on top.

2. **Everything inline in a single tool file** — no separate core module.
   - *Pros:* Fewer files for a small phase-1 feature.
   - *Cons:* Breaks the established core/tools convention; harder to unit test without an
     MCP context; more rework when the embedding phase inevitably adds logic here.

3. **Extend `core/parsers.py` itself** with unstructured-based extraction alongside
   `extract_text`.
   - *Pros:* Single parsing module.
   - *Cons:* `parsers.py` is pypdf-based, returns `(page_number, text)` tuples, and is
     scoped to `store_document`'s pdf/txt/md support — bolting on a completely different
     return shape (categorized elements, base64 images, HTML tables) and extension set
     muddies its documented single responsibility.

## Chosen Approach

**#1** — new core module + new tool file, consistent with the existing codebase split.

- `src/mcp_vectordb/core/document_embedding.py`
  - `MULTIMODAL_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".docx", ".pptx"}` — intentionally
    separate from `core/parsers.py::SUPPORTED_EXTENSIONS`, which stays scoped to
    `store_document`.
  - `ExtractedElement` (NamedTuple or pydantic model): `category` (`"text" | "table" |
    "image"`), `element_type` (raw `unstructured` class name, e.g. `"NarrativeText"`,
    `"Table"`, `"Image"`), `page_number: Optional[int]`, `content: str` (plain text for
    text, HTML for tables, base64 for images).
  - `MultimodalExtractionResult`: `text_elements`, `table_elements`, `image_elements`
    (each `List[ExtractedElement]`).
  - `extract_multimodal(file_path: str) -> MultimodalExtractionResult`: calls
    `unstructured.partition.auto.partition(filename=file_path, infer_table_structure=True,
    extract_image_block_types=["Image"], extract_image_block_to_payload=True)`, then maps
    each returned element via its `.category` attribute (`"Table"` → table,
    `"Image"` → image, everything else → text) into the three lists.
  - A `DocumentEmbeddingParseError` exception (mirrors `DocumentParseError` in
    `core/parsers.py`) wraps any unstructured parsing failure.

- `src/mcp_vectordb/tools/document_embedding.py`
  - `@mcp.tool() async def generate_document_embedding(file_path, ctx=None) -> str`:
    validates the file (`validate_file_path` with `MULTIMODAL_SUPPORTED_EXTENSIONS` and
    `settings.document.max_file_size_mb`), calls `extract_multimodal`, logs full per-element
    detail via `logger.debug` (matching `store_document`'s per-chunk debug logging), and
    returns a summary string with counts per category, e.g.:
    ```
    Successfully extracted document 'report.pdf'
    Text chunks: 12
    Table chunks: 2
    Image chunks: 3
    ```

## Risks & Open Questions

- `unstructured`'s PDF/DOCX/PPTX extras pull in system-level dependencies (`poppler`,
  possibly `libreoffice`) not currently present in this project's environment — needs
  verifying/installing at implementation time.
- `unstructured` is a heavy dependency (pulls in `pdfminer.six`, `python-docx`,
  `python-pptx`, `pillow`, NLTK data, etc.) — adds install size and first-import latency.
- Debug logs (`logger.debug`) of full element detail (base64 image payloads) can be large
  if debug logging is enabled — acceptable since it's opt-in, same tradeoff
  `store_document` already makes.
- **Confirmed during implementation:** requesting image extraction
  (`extract_image_block_types=["Image"]`) on a PDF makes `unstructured` auto-resolve to
  the `hi_res` strategy (a layout-detection model), regardless of the `strategy` kwarg
  being omitted. This loads model weights on first use (one-time latency/download the
  first time the tool runs in a given environment) and requires `tesseract` on `PATH`.
- **Confirmed during implementation:** `unstructured` 0.24.1 has no built-in image
  extraction for DOCX/PPTX (see `spec.md`'s Known Limitation section) — accepted as-is
  rather than writing custom `register_picture_partitioner` plugins. `image_elements` is
  always empty for `.docx`/`.pptx` inputs.
- **Confirmed during implementation:** running `hi_res` PDF extraction requires the
  `tesseract` OCR binary on `PATH` in addition to `poppler` — not part of this project's
  documented setup previously; needs adding to environment/README setup instructions.

## Lifecycle / Removal Plan

N/A — purely additive; no existing code path is modified.
