"""In-memory BM25 (Okapi) keyword search over a document corpus."""

import math
import re
from typing import Dict, List, Tuple

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokenizer used for both indexing and querying."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """BM25 Okapi ranking over a fixed corpus of (doc_id, text) pairs."""

    def __init__(self, corpus: List[Tuple[str, str]], k1: float = 1.5, b: float = 0.75):
        """Build the index.

        Args:
            corpus: List of (doc_id, text) pairs to index. Duplicate doc_ids
                overwrite earlier entries.
            k1: Term-frequency saturation parameter.
            b: Length-normalization parameter.
        """
        self.k1 = k1
        self.b = b

        self.doc_ids: List[str] = []
        self._doc_term_freqs: List[Dict[str, int]] = []
        self._doc_lengths: List[int] = []

        for doc_id, text in corpus:
            terms = tokenize(text)
            term_freqs: Dict[str, int] = {}
            for term in terms:
                term_freqs[term] = term_freqs.get(term, 0) + 1
            self.doc_ids.append(doc_id)
            self._doc_term_freqs.append(term_freqs)
            self._doc_lengths.append(len(terms))

        self.num_docs = len(self.doc_ids)
        self.avg_doc_length = (
            sum(self._doc_lengths) / self.num_docs if self.num_docs else 0.0
        )

        # Document frequency per term, for IDF.
        self._doc_freq: Dict[str, int] = {}
        for term_freqs in self._doc_term_freqs:
            for term in term_freqs:
                self._doc_freq[term] = self._doc_freq.get(term, 0) + 1

        self._idf: Dict[str, float] = {
            term: math.log(1 + (self.num_docs - df + 0.5) / (df + 0.5))
            for term, df in self._doc_freq.items()
        }

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """Return up to ``top_k`` (doc_id, score) pairs ranked by BM25 score.

        Documents with a zero score (no query term overlap) are excluded.
        """
        if self.num_docs == 0:
            return []

        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores = [0.0] * self.num_docs
        for term in query_terms:
            idf = self._idf.get(term)
            if idf is None:
                continue
            for i in range(self.num_docs):
                term_freqs = self._doc_term_freqs[i]
                freq = term_freqs.get(term)
                if not freq:
                    continue
                doc_len = self._doc_lengths[i]
                denom = freq + self.k1 * (
                    1 - self.b + self.b * (doc_len / self.avg_doc_length if self.avg_doc_length else 0)
                )
                scores[i] += idf * (freq * (self.k1 + 1)) / denom

        ranked = [
            (self.doc_ids[i], scores[i]) for i in range(self.num_docs) if scores[i] > 0
        ]
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked[:top_k]
