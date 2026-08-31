"""Multimodal document extraction, embedding, and storage tool for the MCP
Vector Database Server.

Implements `generate_document_embedding` as a 3-step orchestrator:

1. Chunking: `process_document` (see `chunking/process_document.py`) parses
   the document with `unstructured` into title-delimited chunks, each
   carrying text plus any table HTML / image base64 found under it.
2. Embedding: every chunk's text is embedded via the configured embedding
   service.
3. Storage: the embedded chunks are stored in the vector database as
   `Document`s, tagged with a shared `document_id` and `has_table`/
   `has_image` metadata flags.

Table HTML and image base64 payloads themselves are not stored anywhere by
this tool (that lives in a separate store, out of scope here) — only the
flags indicating their presence are kept, so a caller can decide whether to
look them up elsewhere.

Re-ingesting the same file is idempotent by content: every chunk is tagged
with a `content_hash` (sha256 of the file's raw bytes), computed and checked
against the target collection before any chunking/embedding work happens.
A repeat call with unchanged content skips re-ingestion entirely rather than
duplicating the corpus (which previously skewed BM25 IDF and let duplicates
crowd out top-k results); pass `force=True` to replace the existing chunks
instead of skipping.
"""

import hashlib
import logging
import os
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, List, Optional, Tuple

from mcp.server.fastmcp import Context
from pydantic import Field

from ..server import mcp
from ..services import get_vector_db, get_embedding_service
from ..config.config import get_settings
from ..models.document import Document
from ..chunking.process_document import (
    DocumentEmbeddingParseError,
    MULTIMODAL_SUPPORTED_EXTENSIONS,
    process_document,
)
from .search import invalidate_bm25_cache
from ..utils.validation import (
    validate_file_path,
    validate_collection_name,
    validate_metadata,
)
from ..utils.exceptions import ValidationError, VectorDBError

logger = logging.getLogger(__name__)

_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def _hash_file(file_path: str) -> str:
    """sha256 of `file_path`'s raw bytes, read in chunks to bound memory use."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(_HASH_CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_pages_failed_note(pages_failed: List[Tuple[int, int]]) -> str:
    """Trailing note appended to the tool's return string when one or more
    PDF page ranges failed to parse — makes a partial ingest visible to the
    caller instead of a plain "Successfully processed" that hides it."""
    if not pages_failed:
        return ""
    ranges = ", ".join(f"{start}-{end}" for start, end in pages_failed)
    return (
        f"\nWARNING: {len(pages_failed)} page range(s) failed to parse and were "
        f"skipped: {ranges}. This document is PARTIALLY indexed."
    )


@mcp.tool()
async def generate_document_embedding(
    file_path: Annotated[
        str, Field(description="Path to the document to process (.pdf, .md, .docx, .pptx)")
    ],
    collection: Annotated[
        str, Field(description="Collection name to store the embedded chunks in")
    ] = "documents",
    metadata: Annotated[
        Optional[Dict[str, Any]], Field(description="Optional metadata to attach to every stored chunk")
    ] = None,
    force: Annotated[
        bool,
        Field(description="If the same file content is already stored in this collection, replace it instead of skipping"),
    ] = False,
    ctx: Context = None,
) -> str:
    """Extract, embed, and store a document's chunks in the vector database.

    Ingests a document by extracting, chunking, embedding, and
    storing its contents into the vector database. Parses files
    using unstructured into title-delimited chunks and tags each
    chunk with metadata indicating whether it contains tables or
    images (has_table, has_image).
    Note: Raw table HTML and base64 image data are not retained.

    Args:
        file_path: Path to the document to process (.pdf, .md, .docx, .pptx)
        collection: Collection name to store the embedded chunks in (default: "documents")
        metadata: Optional metadata to attach to every stored chunk
        force: If the same file content is already stored in this collection,
            replace it instead of skipping (default: False, skip)
        ctx: FastMCP context for logging and progress reporting

    Returns:
        A success message containing the document ID, total chunk count, and
        a modality breakdown (text/table/image chunk counts — a chunk counts
        toward more than one bucket if it carries more than one modality),
        or a message noting the document was already ingested and skipped.
    """
    try:
        settings = get_settings()
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()

        validated_file_path = validate_file_path(
            file_path,
            allowed_extensions=MULTIMODAL_SUPPORTED_EXTENSIONS,
            max_file_size_mb=settings.document.max_file_size_mb,
        )
        validated_collection = validate_collection_name(collection)
        validated_metadata = validate_metadata(metadata) or {}
        source_filename = os.path.basename(validated_file_path)

        if ctx:
            await ctx.info(f"Starting document processing: {validated_file_path}")

        # Idempotency check: hash the raw file bytes and look for an existing
        # ingest of the same content in this collection *before* doing any
        # parsing/embedding work — a repeat call with unchanged content is
        # the common case this guards, so it should be cheap. Re-ingesting
        # without this previously minted a fresh document_id and chunk UUIDs
        # every time, duplicating the corpus and skewing BM25 IDF.
        content_hash = _hash_file(validated_file_path)
        existing_chunks = []
        if await vector_db.collection_exists(validated_collection):
            existing_chunks = await vector_db.get_all_documents(
                validated_collection, filters={"content_hash": content_hash}
            )

        if existing_chunks and not force:
            existing_document_id = existing_chunks[0].metadata.get("document_id", "unknown")
            existing_uploaded_at = existing_chunks[0].metadata.get("uploaded_at", "unknown")
            if ctx:
                await ctx.info(
                    f"'{source_filename}' already ingested as document {existing_document_id}; skipping"
                )
            return (
                f"Document '{source_filename}' already ingested — skipped.\n"
                f"Document ID: {existing_document_id}\n"
                f"Collection: {validated_collection}\n"
                f"Existing chunks: {len(existing_chunks)}\n"
                f"Originally uploaded: {existing_uploaded_at}\n"
                f"Call again with force=True to replace it."
            )

        if existing_chunks and force:
            existing_ids = [chunk.id for chunk in existing_chunks]
            await vector_db.delete_documents(existing_ids, validated_collection)
            invalidate_bm25_cache(validated_collection)
            if ctx:
                await ctx.info(
                    f"Replacing {len(existing_ids)} existing chunk(s) for '{source_filename}' (force=True)"
                )

        # Step 1: chunking. count_tokens/max_tokens size each chunk to the
        # embedding model's actual token budget (see chunking/token_budget.py)
        # so no chunk is silently truncated by the model at embedding time.
        # pages_failed collects any PDF page range that raised during
        # partitioning (see chunking/process_document.py) — previously that
        # failure was only logged, and the tool still reported success with
        # whatever content happened to survive.
        pages_failed: List[Tuple[int, int]] = []
        chunks = process_document(
            validated_file_path,
            parser=settings.document.parser,
            count_tokens=embedding_service.count_tokens,
            max_tokens=embedding_service.max_input_tokens,
            pages_failed=pages_failed,
        )
        total_chunks = len(chunks)
        text_chunk_count = sum(1 for c in chunks if c.text.strip())
        table_chunk_count = sum(1 for c in chunks if c.tableHTML)
        image_chunk_count = sum(1 for c in chunks if c.imageBase64)
        pages_failed_note = _format_pages_failed_note(pages_failed)

        if ctx:
            await ctx.debug(f"Extracted {total_chunks} chunk(s)")
            if pages_failed:
                await ctx.error(
                    f"'{source_filename}': {len(pages_failed)} PDF page range(s) failed to parse: {pages_failed}"
                )

        if not chunks:
            if ctx:
                await ctx.info(f"No chunks extracted from '{source_filename}'")
            status = "FAILED to process" if pages_failed else "Processed"
            return (
                f"{status} document '{source_filename}'\n"
                f"Chunks extracted: 0\n"
                f"Stored document IDs: 0"
                f"{pages_failed_note}"
            )

        document_id = str(uuid.uuid4())
        uploaded_at = datetime.utcnow().isoformat()

        # Check if collection exists, create if needed
        if not await vector_db.collection_exists(validated_collection):
            dimension = embedding_service.dimension
            await vector_db.create_collection(
                name=validated_collection,
                dimension=dimension,
                metadata={"auto_created": True}
            )
            if ctx:
                await ctx.info(f"Auto-created collection: {validated_collection}")

        # Step 2: embedding
        if ctx:
            await ctx.debug(f"Generating embeddings for {total_chunks} chunk(s)")
        embeddings = await embedding_service.generate_embeddings(
            [chunk.text for chunk in chunks]
        )

        # Step 3: storage
        documents = []
        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            chunk_metadata = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "content_hash": content_hash,
                "source_filename": source_filename,
                "file_type": os.path.splitext(validated_file_path)[1].lower(),
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "uploaded_at": uploaded_at,
                "has_table": bool(chunk.tableHTML),
                "has_image": bool(chunk.imageBase64),
                **validated_metadata,
            }
            documents.append(Document(
                id=chunk_id,
                text=chunk.text,
                embedding=embedding,
                metadata=chunk_metadata,
            ))

        doc_ids = await vector_db.store_documents(documents, validated_collection)
        invalidate_bm25_cache(validated_collection)

        status = "Partially processed" if pages_failed else "Successfully processed"
        if ctx:
            await ctx.info(
                f"{status} document '{source_filename}' as {len(doc_ids)} chunk(s)"
            )

        return (
            f"{status} document '{source_filename}'\n"
            f"Document ID: {document_id}\n"
            f"Collection: {validated_collection}\n"
            f"Chunks extracted: {total_chunks}\n"
            f"Text chunks: {text_chunk_count}\n"
            f"Table chunks: {table_chunk_count}\n"
            f"Image chunks: {image_chunk_count}\n"
            f"Stored document IDs: {len(doc_ids)}"
            f"{pages_failed_note}"
        )

    except ValidationError as e:
        error_msg = f"Validation error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)

    except DocumentEmbeddingParseError as e:
        error_msg = f"Document parse error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)

    except VectorDBError as e:
        error_msg = f"Storage error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)

    except ValueError as e:
        # Catches UnsupportedDocumentTypeError (a ValueError subclass) and any
        # other plain ValueError raised below this point.
        error_msg = f"Validation error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)
