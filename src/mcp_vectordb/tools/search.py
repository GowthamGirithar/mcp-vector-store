"""Search tools for the MCP Vector Database Server."""

from typing import Any, Dict, List, Optional, Tuple
from mcp.server.fastmcp.server import Context

from ..server import mcp
from ..services import get_vector_db, get_embedding_service
from ..config.config import get_settings
from ..utils.validation import validate_text, validate_collection_name, validate_top_k
from ..utils.exceptions import VectorDBError, ValidationError
from ..core.bm25 import BM25Index
from ..core.fusion import reciprocal_rank_fusion
from ..core.reranker import rerank as cross_encoder_rerank

# collection -> (document_count, BM25Index); rebuilt whenever the count changes.
_bm25_cache: Dict[str, Tuple[int, BM25Index]] = {}


async def _get_bm25_index(
    vector_db, collection: str, filters: Optional[Dict[str, Any]] = None
) -> BM25Index:
    """Get a BM25 index for a collection, rebuilding it if stale.

    Filtered requests always build a fresh, filter-scoped index rather than
    using the whole-collection cache, so BM25 hits stay consistent with the
    metadata filters applied to the vector-search leg.
    """
    if filters:
        documents = await vector_db.get_all_documents(collection, filters=filters)
        return BM25Index([(doc.id, doc.text) for doc in documents])

    doc_count = await vector_db.count_documents(collection)
    cached = _bm25_cache.get(collection)
    if cached is not None and cached[0] == doc_count:
        return cached[1]

    documents = await vector_db.get_all_documents(collection)
    index = BM25Index([(doc.id, doc.text) for doc in documents])
    _bm25_cache[collection] = (doc_count, index)
    return index


@mcp.tool()
async def similarity_search(
    query: str,
    collection: str = "documents",
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    include_scores: bool = True,
    include_metadata: bool = True,
    min_score: Optional[float] = None,
    max_text_length: Optional[int] = None,
    ctx: Context = None
) -> str:
    """Perform similarity search to find relevant documents in the vector database.

    Args:
        query: The search query text
        collection: Collection name to search in (default: "documents")
        top_k: Number of results to return (1-100). Falls back to the
            configured SEARCH_DEFAULT_TOP_K when omitted.
        filters: Optional metadata filters to apply
        include_scores: Include similarity scores in results (default: True)
        include_metadata: Include document metadata in results (default: True)
        min_score: Minimum similarity score threshold (0.0-1.0). Falls back to
            the configured SEARCH_DEFAULT_MIN_SCORE when omitted.
        max_text_length: Truncate returned chunk text to this many characters (default: None, returns full text)
        ctx: FastMCP context for logging and progress reporting

    Returns:
        Formatted search results
    """
    try:
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()
        search_config = get_settings().search

        if top_k is None:
            top_k = search_config.default_top_k
        if min_score is None:
            min_score = search_config.default_min_score

        # Validate inputs
        validated_query = validate_text(query)
        validated_collection = validate_collection_name(collection)
        validated_top_k = validate_top_k(top_k, max_k=100)

        if min_score is not None:
            if not isinstance(min_score, (int, float)) or not (0.0 <= min_score <= 1.0):
                raise ValidationError("min_score must be a number between 0.0 and 1.0")

        if max_text_length is not None:
            if not isinstance(max_text_length, int) or max_text_length <= 0:
                raise ValidationError("max_text_length must be a positive integer")
        
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
            filters=filters
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
            
            text_preview = result.document.text
            if max_text_length is not None and len(text_preview) > max_text_length:
                text_preview = text_preview[:max_text_length] + "..."
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


@mcp.tool()
async def hybrid_search(
    query: str,
    collection: str = "documents",
    top_k: Optional[int] = None,
    filters: Optional[Dict[str, Any]] = None,
    min_score: Optional[float] = None,
    use_reranker: bool = False,
    max_text_length: Optional[int] = None,
    ctx: Context = None
) -> str:
    """Hybrid search: fuse vector (cosine) similarity and BM25 keyword search via RRF.

    Pipeline: the query is run against both a vector similarity search and an
    in-memory BM25 keyword search over the same collection, the two ranked
    lists are combined with Reciprocal Rank Fusion (RRF), and an optional
    cross-encoder reranker re-scores the fused top candidates. RRF weights/k
    and the reranker model are deployment-level tuning knobs configured via
    SEARCH_RRF_K / SEARCH_VECTOR_WEIGHT / SEARCH_BM25_WEIGHT / SEARCH_RERANKER_MODEL,
    not exposed here.

    Args:
        query: The search query text
        collection: Collection name to search in (default: "documents")
        top_k: Number of results to return (1-100). Falls back to the
            configured SEARCH_DEFAULT_TOP_K when omitted.
        filters: Optional metadata filters applied to both retrieval legs
        min_score: Minimum fused/rerank score threshold. Falls back to the
            configured SEARCH_DEFAULT_MIN_SCORE when omitted.
        use_reranker: Rerank the fused top candidates with a cross-encoder model (default: False)
        max_text_length: Truncate returned chunk text to this many characters (default: None, returns full text)
        ctx: FastMCP context for logging and progress reporting

    Returns:
        Formatted search results
    """
    try:
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()
        search_config = get_settings().search

        if top_k is None:
            top_k = search_config.default_top_k
        if min_score is None:
            min_score = search_config.default_min_score

        validated_query = validate_text(query)
        validated_collection = validate_collection_name(collection)
        validated_top_k = validate_top_k(top_k, max_k=100)

        if max_text_length is not None:
            if not isinstance(max_text_length, int) or max_text_length <= 0:
                raise ValidationError("max_text_length must be a positive integer")

        if min_score is not None:
            if not isinstance(min_score, (int, float)) or not (0.0 <= min_score <= 1.0):
                raise ValidationError("min_score must be a number between 0.0 and 1.0")

        if ctx:
            await ctx.info(f"Performing hybrid search in collection: {validated_collection}")

        if not await vector_db.collection_exists(validated_collection):
            return f"Collection '{validated_collection}' does not exist"

        # Retrieve a wider candidate pool than top_k so RRF (and the optional
        # reranker) has enough signal to work with before the final cut.
        candidate_k = max(validated_top_k * 4, 50)

        if ctx:
            await ctx.debug("Generating embedding for query")
        query_embedding = await embedding_service.generate_embedding(validated_query)

        if ctx:
            await ctx.debug("Running vector similarity search")
        vector_results = await vector_db.similarity_search(
            query_embedding=query_embedding,
            collection=validated_collection,
            top_k=candidate_k,
            filters=filters
        )
        vector_ranking = [r.document.id for r in vector_results]
        documents_by_id = {r.document.id: r.document for r in vector_results}

        if ctx:
            await ctx.debug("Running BM25 keyword search")
        bm25_index = await _get_bm25_index(vector_db, validated_collection, filters=filters)
        bm25_matches = bm25_index.search(validated_query, top_k=candidate_k)
        bm25_ranking = [doc_id for doc_id, _ in bm25_matches]
        bm25_scores = dict(bm25_matches)

        # Fetch full documents for BM25 hits not already returned by vector search.
        missing_ids = [doc_id for doc_id in bm25_ranking if doc_id not in documents_by_id]
        for doc_id in missing_ids:
            doc = await vector_db.get_document(doc_id, validated_collection)
            if doc is not None:
                documents_by_id[doc_id] = doc

        fused = reciprocal_rank_fusion(
            [vector_ranking, bm25_ranking],
            k=search_config.rrf_k,
            weights=[search_config.vector_weight, search_config.bm25_weight]
        )
        fused = [(doc_id, score) for doc_id, score in fused if doc_id in documents_by_id]

        rerank_scores: Dict[str, float] = {}
        if use_reranker and fused:
            if ctx:
                await ctx.debug("Reranking fused candidates with cross-encoder")
            rerank_pool = fused[:candidate_k]
            candidates = [(doc_id, documents_by_id[doc_id].text) for doc_id, _ in rerank_pool]
            reranked = await cross_encoder_rerank(
                validated_query, candidates, model_name=search_config.reranker_model
            )
            rerank_scores = dict(reranked)
            fused = sorted(rerank_pool, key=lambda item: rerank_scores.get(item[0], float("-inf")), reverse=True)

        vector_scores = {r.document.id: r.score for r in vector_results}

        # min_score is applied against vector (cosine) similarity, matching the
        # 0.0-1.0 semantics of similarity_search's min_score. BM25-only hits have
        # no comparable bounded score, so they aren't filtered by this threshold.
        if min_score is not None:
            fused = [
                (doc_id, score) for doc_id, score in fused
                if vector_scores.get(doc_id, 1.0) >= min_score
            ]

        top_results = fused[:validated_top_k]

        if ctx:
            await ctx.info(f"Found {len(top_results)} results for query")

        if not top_results:
            return f"No results found for query: '{validated_query}'"

        response_lines = [
            f"Found {len(top_results)} results for query: '{validated_query}'",
            f"Collection: {validated_collection}",
            f"Pipeline: vector (cosine) + BM25 -> RRF fusion"
            + (" -> cross-encoder rerank" if use_reranker else ""),
            ""
        ]

        for i, (doc_id, fused_score) in enumerate(top_results, 1):
            document = documents_by_id[doc_id]
            response_lines.append(f"Result {i}:")
            response_lines.append(f"  Document ID: {doc_id}")
            if use_reranker and doc_id in rerank_scores:
                response_lines.append(f"  Rerank Score: {rerank_scores[doc_id]:.4f}")
            response_lines.append(f"  Fused RRF Score: {fused_score:.4f}")
            if doc_id in vector_scores:
                response_lines.append(f"  Vector Score: {vector_scores[doc_id]:.4f}")
            if doc_id in bm25_scores:
                response_lines.append(f"  BM25 Score: {bm25_scores[doc_id]:.4f}")

            text_preview = document.text
            if max_text_length is not None and len(text_preview) > max_text_length:
                text_preview = text_preview[:max_text_length] + "..."
            response_lines.append(f"  Text: {text_preview}")

            if document.metadata:
                display_metadata = {
                    k: v for k, v in document.metadata.items()
                    if not k.startswith(('document_id', 'indexed_at', 'version'))
                }
                if display_metadata:
                    response_lines.append(f"  Metadata: {display_metadata}")

            response_lines.append(f"  Created: {document.created_at.isoformat()}")
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