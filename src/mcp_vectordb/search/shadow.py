"""Shadow-mode comparison of TurboQuant-based hybrid search against the real
(exact-vector) hybrid search pipeline.

Runs alongside the real `hybrid_search` request, never blocks or influences
its response, and only logs a comparison. This validates whether TurboQuant's
approximate ranking — fused directly with BM25, with no exact rerank stage —
is good enough to eventually replace the real semantic leg.
"""

import logging
from typing import Dict, List, Sequence

from ..config.config import CompressionConfig
from .fusion import reciprocal_rank_fusion
from .turboquant_scan import scan_top_n
from ..embedding.quantization import TurboQuantCodec

try:
    from langsmith import traceable
except ImportError:
    def traceable(*_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator

logger = logging.getLogger(__name__)


def _overlap_at_k(real_ranking: Sequence[str], shadow_ranking: Sequence[str]) -> float:
    if not real_ranking:
        return 1.0 if not shadow_ranking else 0.0
    real_set = set(real_ranking)
    shadow_set = set(shadow_ranking)
    return len(real_set & shadow_set) / len(real_set)


@traceable(run_type="chain", name="shadow_turboquant_hybrid_search")
async def run_shadow_hybrid_search(
    *,
    query: str,
    query_embedding: List[float],
    bm25_ranking: List[str],
    real_final_ranking: List[str],
    vector_db,
    collection: str,
    filters: Dict,
    compression_config: CompressionConfig,
    rrf_k: int,
    vector_weight: float,
    bm25_weight: float,
    top_k: int,
) -> None:
    """Run the TurboQuant-only shadow pipeline and log a comparison against
    the real pipeline's final ranking. Never raises — a shadow failure must
    never be visible to (or delay) the real request.
    """
    try:
        if not hasattr(vector_db, "get_all_quantized_vectors"):
            logger.info(
                "shadow_turboquant query=%r collection=%r skipped: adapter has no "
                "TurboQuant code storage support",
                query, collection,
            )
            return

        codec = TurboQuantCodec(
            dimension=len(query_embedding),
            bits=compression_config.bits,
            seed=compression_config.seed,
        )

        entries = await vector_db.get_all_quantized_vectors(collection, filters=filters)
        if not entries:
            logger.info(
                "shadow_turboquant query=%r collection=%r skipped: no stored codes",
                query, collection,
            )
            return

        candidate_pool_size = compression_config.candidate_pool_size
        semantic_ranking = scan_top_n(query_embedding, entries, codec, n=candidate_pool_size)

        fused = reciprocal_rank_fusion(
            [semantic_ranking, bm25_ranking],
            k=rrf_k,
            weights=[vector_weight, bm25_weight],
        )
        shadow_final_ranking = [doc_id for doc_id, _ in fused[:top_k]]

        overlap = _overlap_at_k(real_final_ranking, shadow_final_ranking)

        logger.info(
            "shadow_turboquant query=%r collection=%r overlap_at_k=%.3f "
            "real=%r shadow=%r",
            query, collection, overlap, real_final_ranking, shadow_final_ranking,
        )

    except Exception:
        logger.exception(
            "shadow_turboquant query=%r collection=%r failed", query, collection
        )
