"""In-memory BM25 keyword search over a document corpus, backed by bm25s."""

import logging
from typing import List, Tuple

import bm25s

# bm25s hardcodes its own logger to DEBUG on import, independent of the
# app's logging config, and logs on every index build - silence it so a
# rebuild-on-every-write index doesn't spam DEBUG-level logs.
logging.getLogger("bm25s").setLevel(logging.WARNING)

# No stopword removal / stemming: hybrid search relies on BM25 to catch exact
# keyword and identifier matches (error codes, SKUs, etc.) that a stopword
# list or stemmer could otherwise drop or distort.
_TOKENIZE_KWARGS = dict(stopwords=None, return_ids=False, show_progress=False, lower=True)


def tokenize(text: str) -> List[str]:
    return bm25s.tokenize(text, **_TOKENIZE_KWARGS)[0]


class BM25Index:
    """BM25 ranking over a fixed corpus of (doc_id, text) pairs."""

    def __init__(self, corpus: List[Tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        """Build the index.

        Args:
            corpus: List of (doc_id, text) pairs to index. Duplicate doc_ids
                overwrite earlier entries.
            k1: Term-frequency saturation parameter.
            b: Length-normalization parameter.
        """
        self.doc_ids: List[str] = [doc_id for doc_id, _ in corpus]
        self.num_docs = len(self.doc_ids)

        self._retriever = None
        if self.num_docs:
            texts = [text for _, text in corpus]
            corpus_tokens = bm25s.tokenize(texts, **_TOKENIZE_KWARGS)
            self._retriever = bm25s.BM25(k1=k1, b=b)
            self._retriever.index(corpus_tokens, show_progress=False)

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Return up to ``top_k`` (doc_id, score) pairs ranked by BM25 score.

        Documents with a zero score (no query term overlap) are excluded.
        """
        if self.num_docs == 0:
            return []

        query_tokens = bm25s.tokenize(query, **_TOKENIZE_KWARGS)
        if not any(query_tokens):
            return []

        k = min(top_k, self.num_docs)
        indices, scores = self._retriever.retrieve(
            query_tokens, k=k, show_progress=False
        )

        return [
            (self.doc_ids[doc_index], float(score))
            for doc_index, score in zip(indices[0], scores[0])
            if score > 0
        ]
