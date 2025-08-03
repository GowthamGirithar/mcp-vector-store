"""Storage tools for the MCP Vector Database Server."""

from typing import Any, Dict, Optional
from mcp.server.fastmcp.server import Context

from ..server import mcp
from ..services import get_vector_db, get_embedding_service
from ..core.document import Document
from ..utils.validation import validate_text, validate_metadata, validate_collection_name
from ..utils.exceptions import VectorDBError, ValidationError


@mcp.tool()
async def store_text(
    text: str,
    collection: str = "documents",
    metadata: Optional[Dict[str, Any]] = None,
    document_id: Optional[str] = None,
    ctx: Context = None
) -> str:
    """Store text documents in the vector database with automatic embedding generation.
    
    Args:
        text: The text content to store
        collection: Collection name to store the document in (default: "documents")
        metadata: Optional metadata for the document
        document_id: Optional custom document ID (auto-generated if not provided)
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
        
        if ctx:
            ctx.info(f"Storing text document in collection: {validated_collection}")
        
        # Check if collection exists, create if needed
        if not await vector_db.collection_exists(validated_collection):
            dimension = embedding_service.dimension
            await vector_db.create_collection(
                name=validated_collection,
                dimension=dimension,
                metadata={"auto_created": True}
            )
            if ctx:
                ctx.info(f"Auto-created collection: {validated_collection}")
        
        # Generate embedding
        if ctx:
            ctx.debug("Generating embedding for text")
        embedding = await embedding_service.generate_embedding(validated_text)
        
        # Create document
        document = Document(
            text=validated_text,
            embedding=embedding,
            metadata=validated_metadata
        )
        
        if document_id:
            document.id = document_id
        
        if document.metadata is None:
            document.metadata = {}
        
        # Store document
        doc_ids = await vector_db.store_documents([document], validated_collection)
        
        if ctx:
            ctx.info(f"Successfully stored document with ID: {doc_ids[0]}")
        
        return (f"Successfully stored document in collection '{validated_collection}'\n"
                f"Document ID: {doc_ids[0]}\n"
                f"Text length: {len(validated_text)} characters\n"
                f"Embedding dimension: {len(embedding)}\n"
                f"Metadata fields: {list(document.metadata.keys()) if document.metadata else 'None'}")
        
    except ValidationError as e:
        error_msg = f"Validation error: {e.message}"
        if ctx:
            ctx.error(error_msg)
        raise ValueError(error_msg)
    
    except VectorDBError as e:
        error_msg = f"Storage error: {e.message}"
        if ctx:
            ctx.error(error_msg)
        raise RuntimeError(error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if ctx:
            ctx.error(error_msg)
        raise RuntimeError(error_msg)