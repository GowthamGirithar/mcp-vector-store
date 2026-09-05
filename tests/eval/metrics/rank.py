"""Deterministic rank-based retrieval metrics: hit rate, MRR, and NDCG.

All three walk the same retrieved-list-vs-ground-truth-set relevance check,
so they stay in one file rather than being split further.
"""

import math
from typing import Dict, List, Optional


def compute_hit_rate_and_mrr(
    retrieved_texts: List[str],
    ground_truth_indices: List[int],
    text_to_index: Dict[str, int],
) -> Dict[str, float]:
    ground_truth_set = set(ground_truth_indices)
    rank_of_first_hit: Optional[int] = None
    for rank, text in enumerate(retrieved_texts, start=1):
        chunk_index = text_to_index.get(text)
        if chunk_index in ground_truth_set:
            rank_of_first_hit = rank
            break
    return {
        "hit": 1.0 if rank_of_first_hit is not None else 0.0,
        "mrr": 1.0 / rank_of_first_hit if rank_of_first_hit is not None else 0.0,
    }


def compute_ndcg(
    retrieved_texts: List[str],
    ground_truth_indices: List[int],
    text_to_index: Dict[str, int],
) -> float:
    """Binary-relevance NDCG@k: rewards ranking *all* ground-truth chunks

    early, not just the first one — catches the case hit_rate/MRR miss where
    a multi-hop question's second relevant chunk never makes the top-k.
    """
    ground_truth_set = set(ground_truth_indices)
    if not ground_truth_set:
        return 0.0

    dcg = 0.0
    for rank, text in enumerate(retrieved_texts, start=1):
        if text_to_index.get(text) in ground_truth_set:
            dcg += 1.0 / math.log2(rank + 1)

    num_relevant = min(len(ground_truth_set), len(retrieved_texts))
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, num_relevant + 1))
    return dcg / idcg if idcg > 0 else 0.0
