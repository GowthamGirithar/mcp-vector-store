"""Search tools for the MCP Vector Database Server."""

from typing import Any, Dict, Optional
from mcp.server.fastmcp.server import Context

from ..server import mcp
from ..services import get_vector_db, get_embedding_service
from ..utils.validation import validate_text, validate_collection_name, validate_top_k
from ..utils.exceptions import VectorDBError, ValidationError


@mcp.tool()
async def similarity_search(
    query: str,
    collection: str = "documents",
    top_k: int = 10,
    filters: Optional[Dict[str, Any]] = None,
    include_scores: bool = True,
    include_metadata: bool = True,
    min_score: Optional[float] = None,
    ctx: Context = None
) -> str:
    """Perform similarity search to find relevant documents in the vector database.
    
    Args:
        query: The search query text
        collection: Collection name to search in (default: "documents")
        top_k: Number of results to return (1-100, default: 10)
        filters: Optional metadata filters to apply
        include_scores: Include similarity scores in results (default: True)
        include_metadata: Include document metadata in results (default: True)
        min_score: Minimum similarity score threshold (0.0-1.0)
        ctx: FastMCP context for logging and progress reporting
        
    Returns:
        Formatted search results
    """
    try:
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()
        
        # Validate inputs
        validated_query = validate_text(query)
        validated_collection = validate_collection_name(collection)
        validated_top_k = validate_top_k(top_k, max_k=100)
        
        if min_score is not None:
            if not isinstance(min_score, (int, float)) or not (0.0 <= min_score <= 1.0):
                raise ValidationError("min_score must be a number between 0.0 and 1.0")
        
        # # context injected by the mcp provides the way for the client to know the status of 
        # the long running task
        # with context Elicitation , we can even ask it back for somecases like no results found, can we use llm like that
        if ctx:
            await ctx.info(f"Performing similarity search in collection: {validated_collection}")
        
        # Check if collection exists
        if not await vector_db.collection_exists(validated_collection):
            return f"Collection '{validated_collection}' does not exist"
        
        # Generate query embedding
        if ctx:
            await ctx.debug("Generating embedding for query")
        query_embedding = await embedding_service.generate_embedding(validated_query)
        
        # Perform similarity search
        results = await vector_db.similarity_search(
            query_embedding=query_embedding,
            collection=validated_collection,
            top_k=validated_top_k,
            filters=filters  # TODO: Implement metadata filtering
        )
        
        # Apply minimum score filter if specified
        if min_score is not None:
            results = [r for r in results if r.score >= min_score]
        
        if ctx:
            await ctx.info(f"Found {len(results)} results for query")
        
        # Format results
        if not results:
            return f"No results found for query: '{validated_query}'"
        
        # Build response text
        response_lines = [
            f"Found {len(results)} results for query: '{validated_query}'",
            f"Collection: {validated_collection}",
            ""
        ]
        
        for i, result in enumerate(results, 1):
            response_lines.append(f"Result {i}:")
            response_lines.append(f"  Document ID: {result.document.id}")
            
            if include_scores:
                response_lines.append(f"  Similarity Score: {result.score:.4f}")
                if result.distance is not None:
                    response_lines.append(f"  Distance: {result.distance:.4f}")
            
            # Truncate text for display
            text_preview = result.document.text
            if len(text_preview) > 200:
                text_preview = text_preview[:200] + "..."
            response_lines.append(f"  Text: {text_preview}")
            
            if include_metadata and result.document.metadata:
                # Filter out system metadata for cleaner display
                display_metadata = {
                    k: v for k, v in result.document.metadata.items()
                    if not k.startswith(('document_id', 'indexed_at', 'version'))
                }
                if display_metadata:
                    response_lines.append(f"  Metadata: {display_metadata}")
            
            response_lines.append(f"  Created: {result.document.created_at.isoformat()}")
            response_lines.append("")
        
        return "\n".join(response_lines)
        
    except ValidationError as e:
        error_msg = f"Validation error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)
    
    except VectorDBError as e:
        error_msg = f"Search error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)
    
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)