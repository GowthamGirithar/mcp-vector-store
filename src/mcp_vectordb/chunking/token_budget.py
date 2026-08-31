"""Fit title-delimited chunks to an embedding model's token budget.

`chunk_by_title` sizes chunks in characters, with no notion of how many
tokens the embedding model that eventually consumes them can actually see.
Two failure modes follow directly from that mismatch:

- **Truncation**: a chunk whose character count is well inside
  ``max_characters`` can still tokenize past the model's ``max_seq_length``
  (dense prose, or a table linearized inline, both tokenize denser than the
  character budget assumes) — everything past the limit is silently dropped
  by the model, not by this code.
- **Fragmentation**: `combine_text_under_n_chars=0` (disabled, to avoid
  merging across heading boundaries) leaves many chunks a handful of tokens
  long, which wastes an embedding call and a vector-DB row per sliver and
  gives BM25 almost nothing to score.

`fit_chunks_to_budget` runs as a second pass, after `chunk_by_title`, using
the embedding service's own tokenizer:

1. Split any chunk over `budget` tokens on sentence boundaries, with a small
   token overlap between pieces so a fact sitting on a sentence boundary
   isn't stranded in only one piece.
2. Merge any chunk under `min_tokens` into its preceding sibling **within the
   same source chunk's run** — call sites pass one `chunk_by_title` group at
   a time (see `process_document.py`), so a merge never crosses a title
   boundary `chunk_by_title` already decided on.

Table HTML and image payloads travel with whichever text piece they were
attached to; a split only divides the chunk's *text*, never its table/image
lists (tables are linearized as a whole — see `table_text.py` — so they are
sized before this pass runs, not split by it).
"""

import re
from typing import Callable, List, Tuple

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?:\n])\s+")


def _split_sentences(text: str) -> List[str]:
    parts = [p for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    return parts or ([text] if text.strip() else [])


def _tail_within(pieces: List[str], count_tokens: Callable[[str], int], budget: int) -> List[str]:
    """Return the longest suffix of `pieces` whose joined token count fits `budget`."""
    tail: List[str] = []
    for piece in reversed(pieces):
        candidate = [piece] + tail
        if count_tokens(" ".join(candidate)) > budget:
            break
        tail = candidate
    return tail


def split_text_to_budget(
    text: str,
    count_tokens: Callable[[str], int],
    budget: int,
    overlap: int = 32,
) -> List[str]:
    """Split `text` into pieces that each tokenize to at most `budget` tokens.

    Splits on sentence boundaries so a piece never cuts a sentence in half;
    falls back to a hard slice only for a single sentence that alone exceeds
    `budget` (e.g. a run-on line with no punctuation).
    """
    if not text.strip() or count_tokens(text) <= budget:
        return [text] if text.strip() else []

    sentences = _split_sentences(text)
    pieces: List[str] = []
    buf: List[str] = []

    for sentence in sentences:
        trial = buf + [sentence]
        if buf and count_tokens(" ".join(trial)) > budget:
            pieces.append(" ".join(buf))
            buf = _tail_within(buf, count_tokens, overlap) + [sentence]
            if count_tokens(" ".join(buf)) > budget:
                # the overlap tail alone plus this sentence still overflows;
                # drop the tail rather than emit an oversize piece
                buf = [sentence]
        else:
            buf = trial

        while buf and count_tokens(" ".join(buf)) > budget:
            # a single sentence longer than `budget` on its own: hard-split it
            oversize = buf.pop()
            pieces.extend(_hard_split(oversize, count_tokens, budget))

    if buf:
        pieces.append(" ".join(buf))

    return [p for p in pieces if p.strip()]


def _hard_split(sentence: str, count_tokens: Callable[[str], int], budget: int) -> List[str]:
    """Last-resort word-boundary split for a single sentence over budget."""
    words = sentence.split()
    if not words:
        return []
    out: List[str] = []
    buf: List[str] = []
    for word in words:
        trial = buf + [word]
        if buf and count_tokens(" ".join(trial)) > budget:
            out.append(" ".join(buf))
            buf = [word]
        else:
            buf = trial
    if buf:
        out.append(" ".join(buf))
    return out


def fit_chunks_to_budget(
    chunks: List[Tuple[str, List[str], List[str]]],
    count_tokens: Callable[[str], int],
    budget: int,
    min_tokens: int = 20,
    overlap: int = 32,
) -> List[Tuple[str, List[str], List[str]]]:
    """Split oversize and merge undersize `(text, table_html, image_base64)` triples.

    `chunks` is one `chunk_by_title` group (all under the same title/run) in
    document order. Splitting happens first so a chunk that is oversize only
    because of an undersize neighbour merged into it can't occur; merging
    then only ever combines pieces already known to fit.
    """
    split: List[Tuple[str, List[str], List[str]]] = []
    for text, table_html, image_base64 in chunks:
        pieces = split_text_to_budget(text, count_tokens, budget, overlap) or [text]
        for i, piece in enumerate(pieces):
            # table/image payloads ride with the first piece of their source
            # chunk only, so a table is never duplicated across split pieces
            split.append((piece, table_html if i == 0 else [], image_base64 if i == 0 else []))

    merged: List[Tuple[str, List[str], List[str]]] = []
    for text, table_html, image_base64 in split:
        if (
            merged
            and count_tokens(text) < min_tokens
            and count_tokens(merged[-1][0] + "\n" + text) <= budget
        ):
            prev_text, prev_tables, prev_images = merged[-1]
            merged[-1] = (prev_text + "\n" + text, prev_tables + table_html, prev_images + image_base64)
        else:
            merged.append((text, table_html, image_base64))

    return [c for c in merged if c[0].strip() or c[1] or c[2]]
