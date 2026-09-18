"""Stage-1 retrieval over TurboQuant-compressed embeddings.

Chroma's native ANN index can only score raw float vectors, so it can't be
asked to search over compressed codes directly. This is a linear scan
instead: for every stored (doc_id, QuantizedVector) pair, score it against
the query with the codec's approximate distance estimator, and keep the
top-N. It's affordable because the codes are small and the score itself is
cheap — see embedding/quantization.py.
"""

import heapq
import logging
from typing import List, Tuple

from ..embedding.quantization import QuantizedVector, TurboQuantCodec

try:
    from langsmith import traceable
except ImportError:
    def traceable(*_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator

logger = logging.getLogger(__name__)


@traceable(run_type="retriever", name="turboquant_scan")
def scan_top_n(
    query_embedding: List[float],
    entries: List[Tuple[str, QuantizedVector]],
    codec: TurboQuantCodec,
    n: int,
) -> List[str]:
    """Rank `entries` by approximate similarity to `query_embedding` and
    return the top-n doc_ids, best first.

    This is stage 1 of the two-stage funnel: its ranking is approximate and
    is expected to be corrected by an exact rerank on the returned shortlist
    before being trusted as a final ranking.
    """
    scored = ((codec.score(query_embedding, quantized), doc_id) for doc_id, quantized in entries)
    top = heapq.nlargest(n, scored, key=lambda pair: pair[0])

    logger.debug(
        "turboquant_scan scanned=%d returned=%d", len(entries), len(top)
    )
    return [doc_id for _, doc_id in top]
