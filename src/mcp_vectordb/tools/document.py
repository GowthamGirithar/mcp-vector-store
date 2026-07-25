"""Document upload tool for the MCP Vector Database Server."""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from mcp.server.fastmcp import Context

from ..server import mcp
from ..services import get_vector_db, get_embedding_service
from ..config.config import get_settings
from ..core.document import Document
from ..core.chunking import recursive_chunk
from ..core.parsers import extract_text, DocumentParseError
from ..utils.validation import (
    validate_collection_name,
    validate_metadata,
    validate_file_path,
    validate_chunk_size,
    validate_chunk_overlap,
)
from ..utils.exceptions import VectorDBError, ValidationError

_ALLOWED_EXTENSIONS = {".pdf", ".txt", ".md"}


@mcp.tool()
async def upload_document(
    file_path: str,
    collection: str = "documents",
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
    ctx: Context = None
) -> str:
    """Upload a document (PDF, TXT, or MD), chunk it, embed the chunks, and store them.

    Args:
        file_path: Path to the document file to upload (.pdf, .txt, .md)
        collection: Collection name to store the chunks in (default: "documents")
        metadata: Optional metadata to attach to every chunk document
        document_id: Optional custom document ID shared by all chunks (auto-generated if not provided)
        chunk_size: Maximum characters per chunk (defaults to configured document.chunk_size)
        chunk_overlap: Overlap characters between hard-window chunks (defaults to configured document.chunk_overlap)
        ctx: FastMCP context for logging and progress reporting

    Returns:
        Success message with upload details
    """
    try:
        settings = get_settings()
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()

        # Resolve chunk_size/chunk_overlap defaults from settings, then validate
        resolved_chunk_size = (
            chunk_size if chunk_size is not None else settings.document.chunk_size
        )
        resolved_chunk_overlap = (
            chunk_overlap if chunk_overlap is not None else settings.document.chunk_overlap
        )
        validated_chunk_size = validate_chunk_size(resolved_chunk_size)
        validated_chunk_overlap = validate_chunk_overlap(
            resolved_chunk_overlap, validated_chunk_size
        )

        # Validate inputs
        validated_file_path = validate_file_path(
            file_path,
            allowed_extensions=_ALLOWED_EXTENSIONS,
            max_file_size_mb=settings.document.max_file_size_mb,
        )
        validated_collection = validate_collection_name(collection)
        validated_metadata = validate_metadata(metadata) or {}

        if ctx:
            await ctx.info(f"Starting document upload: {validated_file_path}")

        # Extract text (page_number, text) pairs
        pages = extract_text(validated_file_path)

        # Chunk each page
        page_chunks = []
        for page_index, (page_number, page_text) in enumerate(pages, start=1):
            chunks = recursive_chunk(
                page_text, validated_chunk_size, validated_chunk_overlap
            )
            if ctx:
                await ctx.debug(
                    f"Processing page {page_index}/{len(pages)}: {len(chunks)} chunk(s)"
                )
            for chunk_text in chunks:
                page_chunks.append((page_number, chunk_text))

        total_chunks = len(page_chunks)
        resolved_document_id = document_id if document_id else str(uuid.uuid4())
        source_filename = os.path.basename(validated_file_path)
        _, file_extension = os.path.splitext(validated_file_path)
        file_extension = file_extension.lower()
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

        # Build one Document per chunk
        documents = []
        for chunk_index, (page_number, chunk_text) in enumerate(page_chunks):
            chunk_metadata = {
                "document_id": resolved_document_id,
                "source_filename": source_filename,
                "file_type": file_extension,
                "page_number": page_number,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "uploaded_at": uploaded_at,
                **validated_metadata,
            }
            documents.append(Document(text=chunk_text, metadata=chunk_metadata))

        # Batch-generate embeddings for all chunks
        if ctx:
            await ctx.debug(f"Generating embeddings for {total_chunks} chunk(s)")
        embeddings = await embedding_service.generate_embeddings(
            [doc.text for doc in documents]
        )
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding

        # Store documents
        doc_ids = await vector_db.store_documents(documents, validated_collection)

        if ctx:
            await ctx.info(
                f"Successfully uploaded document '{source_filename}' as {total_chunks} chunk(s)"
            )

        return (
            f"Successfully uploaded document in collection '{validated_collection}'\n"
            f"Document ID: {resolved_document_id}\n"
            f"Source filename: {source_filename}\n"
            f"Pages processed: {len(pages)}\n"
            f"Total chunks: {total_chunks}\n"
            f"Stored document IDs: {len(doc_ids)}"
        )

    except ValidationError as e:
        error_msg = f"Validation error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)

    except DocumentParseError as e:
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
        # Catches UnsupportedFileTypeError (a ValueError subclass) and any
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
