"""Reciprocal Rank Fusion (RRF) for combining ranked result lists."""

from typing import Dict, List, Sequence, Tuple


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]],
    k: int = 60,
    weights: Sequence[float] = None,
) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists of doc_ids into a single ranking via RRF.

    Each ranked list contributes ``weight / (k + rank)`` to a doc_id's fused
    score, where ``rank`` is its 1-based position in that list. Doc_ids
    missing from a list simply don't receive a contribution from it.

    Args:
        rankings: One ranked list of doc_ids per retrieval leg (best first).
        k: RRF constant that dampens the impact of high ranks (standard: 60).
        weights: Optional per-leg weight, same length as ``rankings``.
            Defaults to 1.0 for every leg.

    Returns:
        (doc_id, fused_score) pairs sorted by fused_score descending.
    """
    if weights is None:
        weights = [1.0] * len(rankings)
    elif len(weights) != len(rankings):
        raise ValueError("weights must have the same length as rankings")

    fused_scores: Dict[str, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, doc_id in enumerate(ranking, start=1):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + weight / (k + rank)

    return sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
