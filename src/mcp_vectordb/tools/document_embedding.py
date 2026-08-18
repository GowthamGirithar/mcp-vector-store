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
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, Optional

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
from ..utils.validation import (
    validate_file_path,
    validate_collection_name,
    validate_metadata,
)
from ..utils.exceptions import ValidationError, VectorDBError

logger = logging.getLogger(__name__)


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
        ctx: FastMCP context for logging and progress reporting

    Returns:
        A success message containing the document ID and total chunk count.
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

        # Step 1: chunking
        chunks = process_document(validated_file_path, parser=settings.document.parser)
        total_chunks = len(chunks)

        if ctx:
            await ctx.debug(f"Extracted {total_chunks} chunk(s)")

        if not chunks:
            if ctx:
                await ctx.info(f"No chunks extracted from '{source_filename}'")
            return (
                f"Processed document '{source_filename}'\n"
                f"Chunks extracted: 0\n"
                f"Stored document IDs: 0"
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

        if ctx:
            await ctx.info(
                f"Successfully processed document '{source_filename}' as {len(doc_ids)} chunk(s)"
            )

        return (
            f"Successfully processed document '{source_filename}'\n"
            f"Document ID: {document_id}\n"
            f"Collection: {validated_collection}\n"
            f"Chunks extracted: {total_chunks}\n"
            f"Stored document IDs: {len(doc_ids)}"
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
