"""Optional cross-encoder reranking stage for hybrid search results."""

import asyncio
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_model_cache = {}


def _get_model(model_name: str):
    """Lazily load and cache a CrossEncoder model by name."""
    if model_name not in _model_cache:
        from sentence_transformers import CrossEncoder

        logger.info(f"Loading reranker model: {model_name}")
        _model_cache[model_name] = CrossEncoder(model_name)
    return _model_cache[model_name]


async def rerank(
    query: str,
    candidates: List[Tuple[str, str]],
    model_name: str = _DEFAULT_MODEL,
) -> List[Tuple[str, float]]:
    """Rerank (doc_id, text) candidates against the query using a cross-encoder.

    Args:
        query: The search query text.
        candidates: (doc_id, text) pairs to score, in any order.
        model_name: Cross-encoder model to use.

    Returns:
        (doc_id, rerank_score) pairs sorted by score descending.
    """
    if not candidates:
        return []

    def _score() -> List[Tuple[str, float]]:
        model = _get_model(model_name)
        pairs = [[query, text] for _, text in candidates]
        scores = model.predict(pairs)
        doc_ids = [doc_id for doc_id, _ in candidates]
        return sorted(zip(doc_ids, (float(s) for s in scores)), key=lambda item: item[1], reverse=True)

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _score)
