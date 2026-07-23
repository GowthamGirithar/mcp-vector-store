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
from ..core.chunking import recursive_chunk, structural_chunk
from ..core.parsers import extract_text, get_file_extension, SUPPORTED_EXTENSIONS, DocumentParseError
from ..utils.validation import (
    validate_text,
    validate_metadata,
    validate_collection_name,
    validate_file_path,
    validate_chunk_strategy,
)
from ..utils.exceptions import VectorDBError, ValidationError

# Default chunk_strategy per file extension, used when the caller doesn't
# pass chunk_strategy explicitly: .md/.pdf have structure worth keying off
# (headings, pages), .txt is flat text with nothing structural to split on.
_DEFAULT_CHUNK_STRATEGY_BY_EXTENSION = {
    ".md": "structural",
    ".pdf": "structural",
    ".txt": "recursive",
}


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
    chunk_strategy: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Upload a document (PDF, TXT, or MD), chunk it, embed the chunks, and store them.

    Args:
        file_path: Path to the document file to upload (.pdf, .txt, .md)
        collection: Collection name to store the chunks in (default: "documents")
        metadata: Optional metadata to attach to every chunk document
        chunk_strategy: Chunking strategy to use, "recursive" or "structural" (defaults to "structural" for .md/.pdf and "recursive" for .txt when not given)
        ctx: FastMCP context for logging and progress reporting

    Returns:
        Success message with upload details
    """
    try:
        settings = get_settings()
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()

        # chunk_size/chunk_overlap are not caller-configurable and are validated
        # once at settings load time (see DocumentConfig._validate_chunking)
        chunk_size = settings.document.chunk_size
        chunk_overlap = settings.document.chunk_overlap

        # Validate inputs
        validated_file_path = validate_file_path(
            file_path,
            allowed_extensions=SUPPORTED_EXTENSIONS,
            max_file_size_mb=settings.document.max_file_size_mb,
        )
        validated_collection = validate_collection_name(collection)
        validated_metadata = validate_metadata(metadata) or {}
        if chunk_strategy is not None:
            validate_chunk_strategy(chunk_strategy)

        file_extension = get_file_extension(validated_file_path)

        # If the caller passes chunk_strategy, honor it (already validated
        # above). Otherwise, decide based on the file's extension rather
        # than a single static config default (see
        # _DEFAULT_CHUNK_STRATEGY_BY_EXTENSION).
        validated_chunk_strategy = (
            chunk_strategy
            if chunk_strategy is not None
            else _DEFAULT_CHUNK_STRATEGY_BY_EXTENSION.get(
                file_extension, settings.document.chunk_strategy
            )
        )

        if ctx:
            await ctx.info(f"Starting document upload: {validated_file_path}")

        # Extract text (page_number, text) pairs
        pages = extract_text(validated_file_path)

        # Chunk the document according to the resolved strategy
        page_chunks = []
        if validated_chunk_strategy == "structural":
            page_chunks = structural_chunk(
                pages, chunk_size, chunk_overlap
            )
            if ctx:
                await ctx.debug(
                    f"Structural chunking produced {len(page_chunks)} chunk(s)"
                )
        else:
            for page_index, (page_number, page_text) in enumerate(pages, start=1):
                chunks = recursive_chunk(
                    page_text, chunk_size, chunk_overlap
                )
                if ctx:
                    await ctx.debug(
                        f"Processing page {page_index}/{len(pages)}: {len(chunks)} chunk(s)"
                    )
                for chunk_text in chunks:
                    page_chunks.append((page_number, chunk_text))

        total_chunks = len(page_chunks)
        resolved_document_id = str(uuid.uuid4())
        source_filename = os.path.basename(validated_file_path)
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
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "uploaded_at": uploaded_at,
                **validated_metadata,
            }
            if page_number is not None:
                chunk_metadata["page_number"] = page_number
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