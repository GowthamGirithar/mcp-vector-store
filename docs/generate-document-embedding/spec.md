# Generate Document Embedding — Spec
**Last updated:** 2026-07-25 · **Model:** claude-sonnet-5 (revised after implementation discovery)

## Background

`store_document` (`src/mcp_vectordb/tools/storage.py`) only handles PDF/TXT/MD through a
custom pypdf-based parser (`core/parsers.py::extract_text`), returning plain text per page.
There is no way to separate out tables or images distinctly, and no support for DOCX/PPTX.

This feature adds a new tool, `generate_document_embedding`, that uses the `unstructured`
Python library to partition an uploaded document into distinct **text**, **table**, and
**image** elements. The name is forward-looking (a future phase will add embedding
generation and storage on top of this); this phase implements extraction only.

## Goals

- New MCP tool `generate_document_embedding(file_path, ...)` that parses a document with
  `unstructured` and separates its content into text, table, and image elements.
- Support `.pdf`, `.md`, `.docx`, `.pptx` initially (scope is extensible to more types later).
- Each extracted element (produced by the underlying core extraction function) carries:
  - `category`: `"text"` | `"table"` | `"image"`
  - the raw `unstructured` element type (e.g. `NarrativeText`, `Table`, `Image`)
  - page/slide number, when the format provides one (`None` otherwise)
  - content: plain text for text elements, HTML (via `infer_table_structure`) for tables,
    base64-encoded bytes (via `extract_image_block_to_payload`, no disk writes) for images
- The MCP tool's returned string is a summary, matching `store_document`'s convention —
  e.g.:
  ```
  Successfully extracted document 'report.pdf'
  Text chunks: 12
  Table chunks: 2
  Image chunks: 3
  ```
  Full per-element detail (including image base64 payloads and table HTML) is logged via
  `logger.debug`, matching `store_document`'s per-chunk debug logging — it is not dumped
  into the tool's response, to avoid bloating the response for image-heavy documents.

## Non-goals

- No embedding generation of any kind (text, table, or image) in this phase.
- No vector DB storage — nothing is written to ChromaDB.
- No chunking/splitting of extracted text (no reuse of `core/chunking.py`'s
  `recursive_chunk`/`chunk_data` in this phase — extraction returns `unstructured`'s
  elements as-is).
- No integration with the existing `store_document`/`store_text` tools.
- No custom image-extraction plugins for DOCX/PPTX (see Known Limitation below) — image
  extraction is scoped to what `unstructured` supports out of the box.

## Known Limitation

`unstructured` 0.24.1 (latest as of this writing) only extracts embedded images out of the
box for **PDF** (via `extract_image_block_types`/`extract_image_block_to_payload`). For
**DOCX** and **PPTX**, image extraction is an intentionally pluggable extension point
(`register_picture_partitioner`) with no built-in implementation shipped — confirmed by
inspecting `unstructured/partition/docx.py` and `pptx.py` source, and empirically verified
against this feature's own test fixtures (Table/Text extraction works correctly for all
four formats; Image elements only appear for PDF).

Decision: accept this as a known limitation rather than writing custom picture-partitioner
plugins. For `.docx` and `.pptx` inputs, `image_elements` is always empty — this is
expected behavior, not a bug, and is covered by the spec scenario below.

## Constraints

- Reuses existing file validation conventions (`validate_file_path`,
  `settings.document.max_file_size_mb`), extended to a new supported-extension set
  (`.pdf`, `.md`, `.docx`, `.pptx`) distinct from `core/parsers.py::SUPPORTED_EXTENSIONS`
  (which remains scoped to `store_document`'s existing pdf/txt/md support).
- Adds `unstructured` (with relevant extras, e.g. `unstructured[pdf,docx,pptx,md]`) as a
  new dependency. Flagged as a risk: some extras pull in system-level dependencies
  (e.g. `poppler` for PDF image extraction, `libreoffice` for some PPTX/DOCX paths) that
  aren't currently part of this project's environment.

## Assumptions

- `unstructured`'s `partition()` auto-dispatches by file extension/content and can produce
  `Table` elements with `metadata.text_as_html` and `Image` elements with
  `metadata.image_base64` when the right partition kwargs are passed
  (`infer_table_structure=True`, `extract_image_block_types=["Image"]`,
  `extract_image_block_to_payload=True`).
- Page/slide numbers are best-effort: PDFs and PPTX reliably provide them via
  `element.metadata.page_number`; DOCX may not always provide one, in which case it's `None`.
- This is a new, standalone MCP tool — it does not modify `store_document`, `store_text`,
  `core/parsers.py`, or `core/chunking.py`.

## Spec

GIVEN a valid PDF containing text, a table, and an image
WHEN `generate_document_embedding` is called with its file path
THEN it parses the document with `unstructured` and internally produces elements split into
  text/table/image categories, each with page number and category-appropriate content, and
  returns a summary string reporting counts per category.

GIVEN a `.md`, `.docx`, or `.pptx` file
WHEN `generate_document_embedding` is called
THEN elements are extracted the same way, with page/slide number populated where the format
  provides one and `None` otherwise.

GIVEN a `.docx` or `.pptx` file containing an embedded image
WHEN `generate_document_embedding` is called
THEN `image_elements` is empty (known limitation — see above) while text and table elements
  are still extracted correctly; this is expected, not an error.

GIVEN a file with an unsupported extension
WHEN `generate_document_embedding` is called
THEN it raises a validation error before attempting to parse the file, listing the
  supported extensions.

GIVEN a corrupt or otherwise unparseable file (of a supported extension)
WHEN `generate_document_embedding` is called
THEN a parse error is raised, distinct from a validation error, mirroring
  `DocumentParseError`'s role for `store_document`.

GIVEN a file exceeding `settings.document.max_file_size_mb`
WHEN `generate_document_embedding` is called
THEN it is rejected before parsing (existing `validate_file_path` behavior, reused as-is).

GIVEN a document with no tables and no images (plain text only)
WHEN `generate_document_embedding` is called
THEN the table and image element lists are empty and the summary reports 0 for those
  categories — no error.

GIVEN a document with no extractable content at all (empty document)
WHEN `generate_document_embedding` is called
THEN all three category counts are 0 and no error is raised.
