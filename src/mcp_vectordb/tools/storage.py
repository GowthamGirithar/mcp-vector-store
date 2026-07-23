"""Storage tools for the MCP Vector Database Server."""

import os
import uuid
from datetime import datetime
from typing import Any, Dict, Optional
from mcp.server.fastmcp import Context

from ..server import mcp
from ..services import get_vector_db, get_embedding_service
from ..config.config import get_settings
from ..core.document import Document
from ..core.chunking import chunk_data
from ..core.parsers import extract_text, get_file_extension, SUPPORTED_EXTENSIONS, DocumentParseError
from ..utils.validation import (
    validate_text,
    validate_metadata,
    validate_collection_name,
    validate_file_path,
)
from ..utils.exceptions import VectorDBError, ValidationError


@mcp.tool()
async def store_text(
    text: str,
    collection: str = "documents",
    metadata: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> str:
    """Store text documents in the vector database with automatic embedding generation.

    Args:
        text: The text content to store
        collection: Collection name to store the document in (default: "documents")
        metadata: Optional metadata for the document
        ctx: FastMCP context for logging and progress reporting
        
    Returns:
        Success message with document details
    """
    try:
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()
        
        # Validate inputs
        validated_text = validate_text(text)
        validated_collection = validate_collection_name(collection)
        validated_metadata = validate_metadata(metadata)
        
        # context injected by the mcp provides the way for the client to know the status of 
        # the long running task
        if ctx:
            await ctx.info(f"Storing text document in collection: {validated_collection}")
        
        # Check if collection exists, create if needed
        if not await vector_db.collection_exists(validated_collection):
            dimension = embedding_service.dimension
            await vector_db.create_collection(
                name=validated_collection,
                dimension=dimension,
                metadata={"auto_created": True}
            )
            if ctx:
               await  ctx.info(f"Auto-created collection: {validated_collection}")
        
        # Generate embedding
        if ctx:
            await ctx.debug("Generating embedding for text")
        embedding = await embedding_service.generate_embedding(validated_text)
        
        # Create document
        document = Document(
            text=validated_text,
            embedding=embedding,
            metadata=validated_metadata
        )
        
        if document.metadata is None:
            document.metadata = {}
        
        # Store document
        doc_ids = await vector_db.store_documents([document], validated_collection)
        
        if ctx:
           await ctx.info(f"Successfully stored document with ID: {doc_ids[0]}")
        
        return (f"Successfully stored document in collection '{validated_collection}'\n"
                f"Document ID: {doc_ids[0]}\n"
                f"Text length: {len(validated_text)} characters\n"
                f"Embedding dimension: {len(embedding)}\n"
                f"Metadata fields: {list(document.metadata.keys()) if document.metadata else 'None'}")
        
    except ValidationError as e:
        error_msg = f"Validation error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)
    
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


@mcp.tool()
async def store_document(
    file_path: str,
    collection: str = "documents",
    metadata: Optional[Dict[str, Any]] = None,
    ctx: Context = None
) -> str:
    """Upload a document (PDF, TXT, or MD), chunk it, embed the chunks, and store them.

    Chunking is fully automatic: sections are split on Markdown headings when
    present (falling back to per-page sections otherwise), oversized sections
    are recursively split with 10% overlap, and each chunk carries a
    breadcrumb metadata trail (file > section > page).

    Args:
        file_path: Path to the document file to upload (.pdf, .txt, .md)
        collection: Collection name to store the chunks in (default: "documents")
        metadata: Optional metadata to attach to every chunk document
        ctx: FastMCP context for logging and progress reporting

    Returns:
        Success message with upload details
    """
    try:
        settings = get_settings()
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()

        # chunk_size is not caller-configurable and is validated once at
        # settings load time (see DocumentConfig._validate_chunking)
        chunk_size = settings.document.chunk_size

        # Validate inputs
        validated_file_path = validate_file_path(
            file_path,
            allowed_extensions=SUPPORTED_EXTENSIONS,
            max_file_size_mb=settings.document.max_file_size_mb,
        )
        validated_collection = validate_collection_name(collection)
        validated_metadata = validate_metadata(metadata) or {}

        file_extension = get_file_extension(validated_file_path)

        if ctx:
            await ctx.info(f"Starting document upload: {validated_file_path}")

        source_filename = os.path.basename(validated_file_path)
        data = extract_text(validated_file_path)
        chunks = chunk_data(data, chunk_size, source_filename)
        if ctx:
            await ctx.debug(f"Chunking produced {len(chunks)} chunk(s)")

        total_chunks = len(chunks)
        resolved_document_id = str(uuid.uuid4())
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
        for chunk_index, chunk in enumerate(chunks):
            chunk_metadata = {
                "document_id": resolved_document_id,
                "source_filename": source_filename,
                "file_type": file_extension,
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "uploaded_at": uploaded_at,
                "breadcrumb": chunk.breadcrumb,
                **validated_metadata,
            }
            if chunk.page_number is not None:
                chunk_metadata["page_number"] = chunk.page_number
            documents.append(Document(text=chunk.text, metadata=chunk_metadata))

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
            f"Pages processed: {len(data)}\n"
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