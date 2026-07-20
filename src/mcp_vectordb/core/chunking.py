"""Recursive text chunking for the document-upload-tool feature.

The chunker tries a sequence of increasingly granular separators to split
text into pieces no longer than ``chunk_size`` characters, preserving
natural document structure (paragraphs, then lines, then words) as
much as possible. Any piece that is still too long after all separator
based splits falls back to a hard character window with overlap, which
is guaranteed to make progress and therefore always terminates.

Empty string input returns ``[]`` (there is no content to chunk). Chunks
are never empty strings.
"""

import re
from typing import List, Optional, Tuple

_PARAGRAPH_SEP = "\n\n"
_LINE_SEP = "\n"
_WORD_SEP = " "

_HEADING_RE = re.compile(r"^#{1,6}\s.*$", re.MULTILINE)


def recursive_chunk(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    Splitting is attempted, in order, on paragraph breaks (``"\\n\\n"``),
    line breaks (``"\\n"``), and word breaks (``" "``). Any resulting
    piece still longer than ``chunk_size`` (i.e. a single word) is split
    using a hard character window with ``chunk_overlap`` characters of
    overlap between consecutive windows.

    Args:
        text: The input text to chunk.
        chunk_size: Maximum number of characters per chunk. Must be > 0.
        chunk_overlap: Number of overlapping characters between
            consecutive hard-window chunks. Must be >= 0 and strictly
            less than ``chunk_size``.

    Returns:
        A list of non-empty chunk strings. Returns ``[]`` for empty
        string input.

    Raises:
        ValueError: If ``chunk_size`` <= 0, ``chunk_overlap`` < 0, or
            ``chunk_overlap`` >= ``chunk_size``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be >= 0, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    if text == "":
        return []

    chunks = _split_recursive(text, chunk_size, chunk_overlap)
    return [c for c in chunks if c != ""]


def _split_recursive(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Recursively split ``text`` on paragraph/line/word separators."""
    if text == "":
        return []

    if len(text) <= chunk_size:
        return [text]

    return _split_by_separator(
        text, chunk_size, chunk_overlap, _PARAGRAPH_SEP, _split_by_line
    )


def _split_by_line(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text != "" else []
    return _split_by_separator(
        text, chunk_size, chunk_overlap, _LINE_SEP, _split_by_word
    )


def _split_by_word(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text] if text != "" else []
    return _split_by_separator(
        text, chunk_size, chunk_overlap, _WORD_SEP, _hard_window_split
    )


def _split_by_separator(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separator: str,
    next_splitter,
) -> List[str]:
    """Split ``text`` on ``separator``, recursing into ``next_splitter`` for
    any piece that is still too long. Pieces are then greedily merged back
    together up to ``chunk_size`` to avoid producing many tiny chunks."""
    pieces = text.split(separator)

    resolved: List[str] = []
    for piece in pieces:
        if piece == "":
            continue
        if len(piece) <= chunk_size:
            resolved.append(piece)
        else:
            resolved.extend(next_splitter(piece, chunk_size, chunk_overlap))

    return _merge_pieces(resolved, chunk_size, separator)


def _merge_pieces(pieces: List[str], chunk_size: int, separator: str) -> List[str]:
    """Greedily merge adjacent small pieces back together (joined by
    ``separator``) so we don't emit more, smaller chunks than necessary."""
    merged: List[str] = []
    current = ""

    for piece in pieces:
        if current == "":
            candidate = piece
        else:
            candidate = current + separator + piece

        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current != "":
                merged.append(current)
            current = piece

    if current != "":
        merged.append(current)

    return merged


def _hard_window_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Fallback splitter: fixed-size character windows with overlap.

    ``chunk_overlap < chunk_size`` is guaranteed by the caller (validated
    in ``recursive_chunk``), so the step size ``chunk_size - chunk_overlap``
    is always >= 1, guaranteeing forward progress and termination even on
    pathological input with no separators at all.
    """
    if text == "":
        return []

    step = chunk_size - chunk_overlap
    chunks: List[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        if end == text_len:
            break
        start += step

    return chunks


def structural_chunk(
    pages: List[Tuple[Optional[int], str]], chunk_size: int, chunk_overlap: int
) -> List[Tuple[Optional[int], str]]:
    """Split ``pages`` into structurally coherent chunks.

    Unlike :func:`recursive_chunk`, this operates on the whole document's
    already-extracted ``(page_number, text)`` pairs (as produced by
    ``core.parsers.extract_text``) rather than a single string, since the
    strategy differs by document shape:

    - Multi-page input (the PDF case, ``len(pages) > 1``): each page is
      passed through unchanged as exactly one output chunk, even if its
      text is longer than ``chunk_size``.
    - Single-page input with ``page_number is None`` (the ``.md``/``.txt``
      case): the text is split on Markdown heading lines
      (``^#{1,6}\\s.*$``, multiline) into sections — each section is a
      heading line plus its body up to the next heading or EOF, emitted
      whole as one chunk with no size-based re-splitting. If there are no
      heading matches, falls back to :func:`recursive_chunk` so headingless
      text (typically ``.txt``) is chunked identically to before.

    Args:
        pages: The document's ``(page_number, text)`` pairs.
        chunk_size: Forwarded to the no-heading ``recursive_chunk``
            fallback; has no effect when structural splitting occurs.
        chunk_overlap: Forwarded to the no-heading ``recursive_chunk``
            fallback; has no effect when structural splitting occurs.

    Returns:
        A list of ``(page_number, text)`` chunk tuples.
    """
    if not pages:
        return []

    if len(pages) > 1:
        return list(pages)

    page_number, text = pages[0]
    if page_number is not None:
        return [(page_number, text)]

    return _split_by_headings(text, chunk_size, chunk_overlap)


def _split_by_headings(
    text: str, chunk_size: int, chunk_overlap: int
) -> List[Tuple[Optional[int], str]]:
    """Split ``text`` into Markdown heading-delimited sections.

    Falls back to :func:`recursive_chunk` when there are no heading matches.
    """
    matches = list(_HEADING_RE.finditer(text))

    if not matches:
        return [(None, piece) for piece in recursive_chunk(text, chunk_size, chunk_overlap)]

    sections: List[Tuple[Optional[int], str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        sections.append((None, text[start:end]))

    return sections
