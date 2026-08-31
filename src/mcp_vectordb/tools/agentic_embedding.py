"""Agentic embedding tool for the MCP Vector Database Server.

Unlike `generate_document_embedding` (fixed-size/title-based chunking via
`unstructured`), this tool lets an LLM decide chunk boundaries per chunk,
guided by a caller-supplied prompt — the prompt also dictates whatever extra
per-chunk fields (title, summary, tags, ...) the LLM should produce, stored
alongside each chunk as metadata. See `core/agentic_chunking.py` for the
LLM orchestration.
"""

import logging
import os
import uuid
from datetime import datetime
from typing import Annotated, Any, Dict, Optional

from mcp.server.fastmcp import Context
from pydantic import Field

from ..server import mcp
from ..services import get_vector_db, get_embedding_service, get_llm_service
from ..config.config import get_settings
from ..chunking.process_document import (
    DocumentEmbeddingParseError,
    UnsupportedDocumentTypeError,
)
from ..core.agentic_chunking import agentic_chunk_text, agentic_chunk_document
from ..models.document import Document
from .search import invalidate_bm25_cache
from ..utils.exceptions import ValidationError, VectorDBError, LLMServiceError

logger = logging.getLogger(__name__)


@mcp.tool()
async def agentic_generate_embedding(
    prompt: Annotated[
        str, Field(description="Instructions guiding how the LLM should chunk and tag the content")
    ],
    file_path: Annotated[
        Optional[str], Field(description="Path to the document to process (.pdf, .md, .docx, .pptx)")
    ] = None,
    text: Annotated[
        Optional[str], Field(description="Raw text to process, used instead of file_path")
    ] = None,
    collection: Annotated[
        str, Field(description="Collection name to store the embedded chunks in")
    ] = "documents",
    metadata: Annotated[
        Optional[Dict[str, Any]], Field(description="Optional metadata to attach to every stored chunk")
    ] = None,
    ctx: Context = None,
) -> str:
    """Chunk, embed, and store content using an LLM to decide chunk boundaries.

    Uses `text` if provided, otherwise reads and extracts `file_path`. The
    LLM splits the content into chunks per `prompt`'s instructions; any
    extra per-chunk fields the prompt asks for (title, summary, tags, ...)
    are stored alongside each chunk as metadata.

    Args:
        prompt: Instructions guiding how the LLM should chunk and tag the content
        file_path: Path to the document to process (.pdf, .md, .docx, .pptx)
        text: Raw text to process, used instead of file_path
        collection: Collection name to store the embedded chunks in (default: "documents")
        metadata: Optional metadata to attach to every stored chunk
        ctx: FastMCP context for logging and progress reporting

    Returns:
        A success message containing the document ID and total chunk count.
    """
    try:
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()
        llm_service = get_llm_service()
        max_input_chars = get_settings().llm.max_input_chars

        if text is not None:
            source_label = "inline_text"
            if ctx:
                await ctx.debug("Running agentic chunking")
            chunks = await agentic_chunk_text(text, prompt, llm_service, max_input_chars)
        else:
            source_label = os.path.basename(file_path)
            if ctx:
                await ctx.info(f"Processing document: {file_path}")
            chunks = await agentic_chunk_document(file_path, prompt, llm_service, max_input_chars)

        total_chunks = len(chunks)

        if not chunks:
            if ctx:
                await ctx.info(f"No chunks produced from '{source_label}'")
            return (
                f"Processed '{source_label}'\n"
                f"Chunks produced: 0\n"
                f"Stored document IDs: 0"
            )

        document_id = str(uuid.uuid4())
        uploaded_at = datetime.utcnow().isoformat()

        # Check if collection exists, create if needed
        if not await vector_db.collection_exists(collection):
            dimension = embedding_service.dimension
            await vector_db.create_collection(
                name=collection,
                dimension=dimension,
                metadata={"auto_created": True}
            )
            if ctx:
                await ctx.info(f"Auto-created collection: {collection}")

        if ctx:
            await ctx.debug(f"Generating embeddings for {total_chunks} chunk(s)")

        embeddings = await embedding_service.generate_embeddings(
            [chunk.text for chunk in chunks]
        )

        documents = []
        for chunk_index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = str(uuid.uuid4())
            # chunk.metadata is whatever shape the caller's `prompt` asked the
            # LLM to produce (title, summary, tags, ...) — namespaced with
            # "llm_" so it can't collide with the fixed keys below.
            llm_metadata = {f"llm_{key}": value for key, value in chunk.metadata.items()}
            chunk_metadata = {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "source": source_label,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "uploaded_at": uploaded_at,
                "agentic": True,
                **llm_metadata,
                **(metadata or {}),
            }
            documents.append(Document(
                id=chunk_id,
                text=chunk.text,
                embedding=embedding,
                metadata=chunk_metadata,
            ))

        doc_ids = await vector_db.store_documents(documents, collection)
        invalidate_bm25_cache(collection)

        if ctx:
            await ctx.info(
                f"Successfully processed '{source_label}' as {len(doc_ids)} chunk(s)"
            )

        return (
            f"Successfully processed '{source_label}'\n"
            f"Document ID: {document_id}\n"
            f"Collection: {collection}\n"
            f"Chunks produced: {total_chunks}\n"
            f"Stored document IDs: {len(doc_ids)}"
        )

    except ValidationError as e:
        error_msg = f"Validation error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)

    except UnsupportedDocumentTypeError as e:
        error_msg = f"Validation error: {e}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)

    except DocumentEmbeddingParseError as e:
        error_msg = f"Document parse error: {e}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)

    except LLMServiceError as e:
        error_msg = f"LLM error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)

    except VectorDBError as e:
        error_msg = f"Storage error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)
