# Document Upload Tool — Spec
**Last updated:** 2026-07-20 · **Model:** claude-sonnet-5

## Background

The server currently exposes two MCP tools: `store_text` (single-text embedding + storage) and `similarity_search`. There is no way to ingest a whole document (e.g. a multi-page PDF) — a caller would have to manually extract, chunk, and call `store_text` once per chunk themselves. This feature adds a dedicated `upload_document` tool that takes a server-local file path, extracts its text, chunks it, embeds each chunk, and stores all chunks in the vector DB with metadata that makes them easy to find and trace back to the source document later (e.g. filter search results by `document_id`, `source_filename`, or `page_number`).

**Extension (2026-07-20):** `upload_document` shipped (T1-T5) with exactly one chunking strategy — `recursive_chunk` (paragraph → line → word → hard-window). Some documents (structured markdown, multi-page PDFs) chunk better when split along their own structure (headings, page breaks) instead of pure character windows, since it keeps each section's content together for retrieval. This extension adds a second, selectable chunking strategy without breaking existing callers.

## Goals & Constraints

**Goals**
- Support ingesting PDF, `.txt`, and `.md` files in one tool call.
- Split extracted text into chunks using recursive/semantic splitting (paragraph/sentence-aware, falling back to fixed-size windows for oversized paragraphs).
- Generate embeddings for all chunks and store them via the existing `VectorDBAdapter.store_documents` batch path.
- Attach rich per-chunk metadata: `document_id` (groups all chunks from one upload), `source_filename`, `file_type`, `page_number` (PDF only), `chunk_index`, `total_chunks`, `uploaded_at`, plus any user-supplied custom metadata merged in.
- Report progress via the existing `ctx` (Context) pattern used in `store_text`/`similarity_search` (e.g. per page/chunk info logs).
- Validate up front (extension, file existence, file size) and fail fast with the same `ValidationError` → `ValueError` / `VectorDBError` → `RuntimeError` conventions already used by the other tools.
- Chunk size/overlap/max file size are configurable via a new `DocumentConfig` (env-var backed, like `VectorDBConfig`/`EmbeddingConfig`), overridable per-call via optional tool parameters.
- **[Extension]** Support a second chunking strategy, `structural`, selectable via a new `chunk_strategy` tool parameter (`"recursive" | "structural"`), defaulting to `"recursive"` for full backward compatibility. Default is also configurable via `DocumentConfig.chunk_strategy` / `DOCUMENT_CHUNK_STRATEGY`.
- **[Extension]** `structural` splits along each file type's native structure: `.md` splits on Markdown heading lines (`#`, `##`, `###`, ...) into one chunk per heading section; `.pdf` uses the existing per-page extraction as-is (one chunk per page); `.txt` has no structural signal, so it silently falls back to `recursive_chunk` behavior.
- **[Extension]** A structural section is kept as a single chunk even if it exceeds `chunk_size` — structural mode prioritizes semantic coherence over the size contract enforced by `recursive_chunk`. This is a deliberate, documented trade-off.

**Constraints**
- Input is a **server-local file path** (not base64 upload) — the tool assumes the file already exists on disk where the server process runs.
- Processing is **synchronous** within the single tool call — no background job/polling infrastructure.
- New parsing dependency: `pypdf` (or `pypdf2`/`pymupdf` — confirmed during implementation) for PDF text extraction. `.txt`/`.md` are read directly, no new dependency needed for those.
- Must integrate with the existing `Document` model, `EmbeddingService.generate_embeddings` (batch), and `VectorDBAdapter.store_documents` — no changes to adapter interfaces.

**Non-goals**
- OCR for scanned/image-only PDFs.
- DOCX or other office formats (left for a future iteration).
- Deduplication of previously-uploaded documents.
- Background/async job processing with a status-polling tool.
- Automatic re-chunking/re-embedding on file change (no file watching).

## Assumptions

- The MCP server process has filesystem read access to whatever path the client passes in (stdio transport is the primary use case here; SSE/HTTP clients would need to reference a path reachable by the server).
- A single tool call handles one file at a time (no batch/directory upload in this iteration).
- "Page" only has meaning for PDFs; `.txt`/`.md` chunks omit `page_number` (or set it to `null`).

## Spec

```
GIVEN a valid PDF file path with multiple pages
WHEN upload_document is called with that path
THEN the tool extracts text per page, chunks it recursively, embeds each chunk,
     stores all chunks in the target collection, and returns a summary
     (document_id, total_chunks, pages processed, collection)

GIVEN a valid .txt or .md file path
WHEN upload_document is called with that path
THEN the tool reads the file content, chunks it (page_number omitted/null),
     embeds and stores all chunks, and returns a summary

GIVEN a file path with an unsupported extension (e.g. .docx, .csv)
WHEN upload_document is called
THEN the tool raises a ValidationError-derived ValueError before attempting to read the file

GIVEN a file path that does not exist or is not readable
WHEN upload_document is called
THEN the tool raises a ValidationError-derived ValueError identifying the missing/unreadable path

GIVEN a file whose size exceeds the configured max_file_size_mb
WHEN upload_document is called
THEN the tool raises a ValidationError-derived ValueError before parsing, stating the limit

GIVEN a PDF that is corrupt or fails to parse
WHEN upload_document is called
THEN the tool raises a RuntimeError wrapping the parsing failure, consistent with existing
     VectorDBError/RuntimeError handling in store_text

GIVEN custom metadata is supplied by the caller
WHEN upload_document is called
THEN the custom metadata is merged into every chunk's metadata, and reserved keys
     (document_id, source_filename, file_type, page_number, chunk_index, total_chunks, uploaded_at)
     cannot be overridden by caller-supplied metadata (same reserved-key validation pattern as validate_metadata)

GIVEN the target collection does not yet exist
WHEN upload_document is called
THEN the collection is auto-created using the embedding service's dimension, same as store_text

GIVEN chunk_size/chunk_overlap are not passed as tool parameters
WHEN upload_document is called
THEN the values fall back to DocumentConfig defaults loaded from environment variables

GIVEN a ctx (Context) is available
WHEN the document is being processed
THEN the tool emits ctx.info/ctx.debug progress updates (e.g. "Processing page 3/10",
     "Embedding chunk 12/40") consistent with the logging style in store_text/similarity_search

GIVEN chunk_strategy is not passed as a tool parameter
WHEN upload_document is called
THEN the value falls back to DocumentConfig.chunk_strategy (default "recursive"),
     same fallback pattern as chunk_size/chunk_overlap

GIVEN chunk_strategy="structural" and a .md file with heading sections
WHEN upload_document is called
THEN the tool produces one chunk per heading section (content between one heading
     and the next, or end of file), regardless of chunk_size

GIVEN chunk_strategy="structural" and a .pdf file
WHEN upload_document is called
THEN the tool produces one chunk per page, regardless of chunk_size (page_number
     metadata set as usual)

GIVEN chunk_strategy="structural" and a .txt file (no structural signal)
WHEN upload_document is called
THEN the tool falls back to the same output recursive_chunk would produce

GIVEN chunk_strategy is set to a value other than "recursive" or "structural"
WHEN upload_document is called
THEN the tool raises a ValidationError-derived ValueError before any parsing,
     naming the invalid value and the allowed set
```
