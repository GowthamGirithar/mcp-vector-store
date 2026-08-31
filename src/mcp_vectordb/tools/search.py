"""Search tools for the MCP Vector Database Server."""

import logging
from typing import Annotated, Any, Dict, List, Optional, Tuple
from mcp.server.fastmcp.server import Context
from pydantic import Field

from ..server import mcp
from ..services import get_vector_db, get_embedding_service
from ..config.config import get_settings
from ..utils.validation import validate_text, validate_collection_name, validate_top_k
from ..utils.exceptions import VectorDBError, ValidationError
from ..search import BM25Index, reciprocal_rank_fusion
from ..search import rerank as cross_encoder_rerank

logger = logging.getLogger(__name__)

# collection -> (document_count, BM25Index). Primary invalidation is explicit:
# any tool that writes to a collection (store/delete/replace) must call
# invalidate_bm25_cache(collection) after the write commits — see
# tools/document_embedding.py, tools/agentic_embedding.py, tools/storage.py.
# The document_count comparison below is only a defensive fallback for a
# write path that forgets to call it; count alone is not sufficient, since an
# update or a same-size delete+insert (e.g. re-ingesting a document with
# force=True) leaves the count unchanged while the underlying text changes.
_bm25_cache: Dict[str, Tuple[int, BM25Index]] = {}


def invalidate_bm25_cache(collection: str) -> None:
    """Drop the cached BM25 index for `collection`.

    Call this after any write (store, delete, or replace) to that collection
    commits, so the next search rebuilds against current content instead of
    serving a stale index. A no-op if nothing is cached for it yet.
    """
    _bm25_cache.pop(collection, None)


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
    query: Annotated[str, Field(description="The search query text")],
    collection: Annotated[
        str, Field(description="Collection name to search in")
    ] = "documents",
    filters: Annotated[
        Optional[Dict[str, Any]], Field(description="Optional metadata filters to apply")
    ] = None,
    ctx: Context = None
) -> List[str]:
    """Perform similarity search to find relevant documents in the vector database.

    Args:
        query: The search query text
        collection: Collection name to search in (default: "documents")
        filters: Optional metadata filters to apply
        ctx: FastMCP context for logging and progress reporting

    Returns:
        List of matching document text content
    """
    try:
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()
        search_config = get_settings().search

        top_k = search_config.default_top_k
        min_score = search_config.default_min_score

        # Validate inputs
        validated_query = validate_text(query)
        validated_collection = validate_collection_name(collection)
        validated_top_k = validate_top_k(top_k, max_k=100)

        logger.info(
            "similarity_search query=%r collection=%r filters=%r min_score=%r top_k=%r",
            validated_query, validated_collection, filters, min_score, validated_top_k
        )

        # # context injected by the mcp provides the way for the client to know the status of
        # the long running task
        # with context Elicitation , we can even ask it back for somecases like no results found, can we use llm like that
        if ctx:
            await ctx.info(f"Performing similarity search in collection: {validated_collection}")

        # Check if collection exists
        if not await vector_db.collection_exists(validated_collection):
            logger.info("similarity_search collection=%r does not exist", validated_collection)
            return []

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

        logger.info(
            "similarity_search query=%r retrieved=%d documents=%r",
            validated_query, len(results),
            [{"id": r.document.id, "score": r.score} for r in results]
        )

        if ctx:
            await ctx.info(f"Found {len(results)} results for query")

        return [result.document.text for result in results]
        
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
    query: Annotated[str, Field(description="The natural language search query.")],
    collection: Annotated[
        str, Field(description="Collection name to search in")
    ] = "documents",
    filters: Annotated[
        Optional[Dict[str, Any]],
        Field(description="Optional key-value metadata filters applied to limit the search scope."),
    ] = None,
    ctx: Context = None
) -> List[str]:
    """Hybrid search: fuse vector (cosine) similarity and BM25 keyword search via RRF.

    Performs hybrid retrieval (dense vector + sparse BM25) fused via Reciprocal Rank Fusion (RRF)
    and refined with an optional cross-encoder reranker

    Args:
        query: The search query text
        collection: Collection name to search in (default: "documents")
        filters: Optional metadata filters applied to both retrieval legs
        ctx: FastMCP context for logging and progress reporting

    Returns:
        List of matching document text content
    """
    try:
        vector_db = get_vector_db()
        embedding_service = get_embedding_service()
        search_config = get_settings().search

        top_k = search_config.default_top_k
        min_score = search_config.default_min_score
        use_reranker = search_config.use_reranker

        validated_query = validate_text(query)
        validated_collection = validate_collection_name(collection)
        validated_top_k = validate_top_k(top_k, max_k=100)

        logger.info(
            "hybrid_search query=%r collection=%r filters=%r min_score=%r top_k=%r use_reranker=%r",
            validated_query, validated_collection, filters, min_score, validated_top_k, use_reranker
        )

        if ctx:
            await ctx.info(f"Performing hybrid search in collection: {validated_collection}")

        if not await vector_db.collection_exists(validated_collection):
            logger.info("hybrid_search collection=%r does not exist", validated_collection)
            return []

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

        logger.info(
            "hybrid_search query=%r vector_db retrieved=%d documents=%r",
            validated_query, len(vector_results),
            [{"id": r.document.id, "score": r.score} for r in vector_results]
        )

        if ctx:
            await ctx.debug("Running BM25 keyword search")
        bm25_index = await _get_bm25_index(vector_db, validated_collection, filters=filters)
        bm25_matches = bm25_index.search(validated_query, top_k=candidate_k)
        bm25_ranking = [doc_id for doc_id, _ in bm25_matches]
        bm25_scores = dict(bm25_matches)

        logger.info(
            "hybrid_search query=%r bm25 retrieved=%d documents=%r",
            validated_query, len(bm25_matches),
            [{"id": doc_id, "score": score} for doc_id, score in bm25_matches]
        )

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

        logger.info(
            "hybrid_search query=%r rrf fused=%d documents=%r",
            validated_query, len(fused),
            [{"id": doc_id, "score": score} for doc_id, score in fused]
        )

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

            logger.info(
                "hybrid_search query=%r rerank reranked=%d documents=%r",
                validated_query, len(fused),
                [{"id": doc_id, "score": rerank_scores.get(doc_id)} for doc_id, _ in fused]
            )

        vector_scores = {r.document.id: r.score for r in vector_results}

        # min_score is a 0.0-1.0 confidence floor. Vector (cosine) scores are
        # already in that range and are checked directly. BM25 scores are
        # unbounded, so a BM25-only hit (no vector score at all) used to
        # default to 1.0 here and pass the threshold unconditionally — a
        # min_score filter that only ever filtered the vector leg is not a
        # min_score filter. Instead, BM25 scores are min-max normalized
        # against the strongest BM25 match in *this* candidate pool (the
        # only scale a raw BM25 score can meaningfully be compared against),
        # and a document passes if either leg clears the bar — consistent
        # with hybrid search's premise that a strong match on either signal
        # is worth keeping.
        if min_score is not None:
            max_bm25_score = max(bm25_scores.values(), default=0.0)

            def _passes_min_score(doc_id: str) -> bool:
                vector_score = vector_scores.get(doc_id)
                if vector_score is not None and vector_score >= min_score:
                    return True
                raw_bm25_score = bm25_scores.get(doc_id)
                if raw_bm25_score is not None and max_bm25_score > 0:
                    return (raw_bm25_score / max_bm25_score) >= min_score
                return False

            fused = [(doc_id, score) for doc_id, score in fused if _passes_min_score(doc_id)]

        top_results = fused[:validated_top_k]

        logger.info(
            "hybrid_search query=%r final=%d documents=%r",
            validated_query, len(top_results),
            [doc_id for doc_id, _ in top_results]
        )

        if ctx:
            await ctx.info(f"Found {len(top_results)} results for query")

        return [documents_by_id[doc_id].text for doc_id, _ in top_results]

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