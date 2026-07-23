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
from typing import List, NamedTuple, Optional, Tuple

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
        text: The input text to chunk. ``chunk_size`` must be > 0 and
            ``chunk_overlap`` must be >= 0 and strictly less than
            ``chunk_size`` — the caller (``store_document``) is
            responsible for this, since these values are sourced from
            config validated at startup (see
            ``DocumentConfig._validate_chunking``).
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between
            consecutive hard-window chunks.

    Returns:
        A list of non-empty chunk strings. Returns ``[]`` for empty
        string input.
    """
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

    ``chunk_overlap < chunk_size`` is assumed to hold (see
    ``recursive_chunk``), so the step size ``chunk_size - chunk_overlap``
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


class Chunk(NamedTuple):
    """One chunk produced by :func:`chunk_data`, ready to store.

    ``breadcrumb`` is the fully-assembled trail (e.g.
    ``"report.md > Intro > Background"`` or ``"sample.pdf > page 2"``) —
    callers don't need to reconstruct it from a heading path/page number.
    """

    page_number: Optional[int]
    text: str
    breadcrumb: str


def chunk_data(
    pages: List[Tuple[Optional[int], str]], chunk_size: int, source_filename: str
) -> List[Chunk]:
    """Chunk a document's extracted pages using the auto-chunking pipeline.

    Operates on the whole document's already-extracted ``(page_number,
    text)`` pairs (as produced by ``core.parsers.extract_text``) through
    five stages:

    1/2. Decide header-based splitting vs layout-aware extraction, and
         split into sections accordingly (see :func:`_split_into_sections`).
    3/4. Split any oversized section with :func:`recursive_chunk`, using
         10% of ``chunk_size`` as overlap; keep smaller sections whole.
    5.   Attach a breadcrumb to every resulting chunk.

    Args:
        pages: The document's ``(page_number, text)`` pairs.
        chunk_size: Maximum section size (also used as the approximate
            token-count ceiling) before a section is recursively split.
        source_filename: The document's filename, used as the breadcrumb's
            root component.

    Returns:
        A list of :class:`Chunk`.
    """
    if not pages:
        return []

    sections = _split_into_sections(pages)
    chunk_overlap = max(1, round(chunk_size * 0.1))

    chunks: List[Chunk] = []
    for page_number, text, heading_path in sections:
        pieces = (
            recursive_chunk(text, chunk_size, chunk_overlap)
            if _approx_token_count(text) > chunk_size
            else [text]
        )
        breadcrumb = _build_breadcrumb(source_filename, heading_path, page_number)
        chunks.extend(Chunk(page_number, piece, breadcrumb) for piece in pieces)

    return chunks


def _split_into_sections(
    pages: List[Tuple[Optional[int], str]]
) -> List[Tuple[Optional[int], str, str]]:
    """Stage 1/2: decide header-based splitting vs layout-aware extraction,
    and split ``pages`` into sections accordingly.

    - Text with ``page_number is None`` (``.md``/``.txt``) that contains
      Markdown headings: header-based splitting — split on heading lines
      (``^#{1,6}\\s.*$``, multiline) into sections, each tagged with its
      heading hierarchy path (e.g. ``"Intro > Background"``).
    - Anything else (PDF pages, or ``.md``/``.txt`` with no headings):
      layout-aware extraction — each page/text is already a section,
      tagged with an empty heading path.

    ``page_number is None`` is a reliable signal for "this is plain/Markdown
    text, not a PDF page": ``core.parsers.extract_text`` always returns
    exactly one ``(None, text)`` tuple for ``.txt``/``.md`` input, and always
    assigns a page number starting at 1 for every PDF page — so there is
    never more than one ``page_number is None`` entry to worry about.
    """
    first_page_number, first_text = pages[0]
    if first_page_number is None and _HEADING_RE.search(first_text):
        return [
            (None, section_text, heading_path)
            for heading_path, section_text in _split_by_headings_with_path(first_text)
        ]

    return _layout_aware_extraction(pages)


def _layout_aware_extraction(
    pages: List[Tuple[Optional[int], str]]
) -> List[Tuple[Optional[int], str, str]]:
    """Treat each already-extracted page/text as one section (no heading
    ancestry), dropping empty pages."""
    return [(page_number, text, "") for page_number, text in pages if text != ""]


def _build_breadcrumb(
    source_filename: str, heading_path: str, page_number: Optional[int]
) -> str:
    """Stage 5: assemble the breadcrumb trail for a chunk."""
    parts = [source_filename]
    if heading_path:
        parts.append(heading_path)
    if page_number is not None:
        parts.append(f"page {page_number}")
    return " > ".join(parts)


def _split_by_headings_with_path(text: str) -> List[Tuple[str, str]]:
    """Split ``text`` into Markdown heading-delimited sections, pairing each
    section with its heading hierarchy path (e.g. ``"Intro > Background"``),
    built by tracking heading level nesting via a stack."""
    matches = list(_HEADING_RE.finditer(text))

    sections: List[Tuple[str, str]] = []
    stack: List[Tuple[int, str]] = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)

        heading_line = match.group().strip()
        level = len(heading_line) - len(heading_line.lstrip("#"))
        title = heading_line.lstrip("#").strip()

        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))

        heading_path = " > ".join(t for _, t in stack)
        sections.append((heading_path, text[start:end]))

    return sections


def _approx_token_count(text: str) -> float:
    """Approximate the token count of ``text`` as ``word count * 1.3``."""
    return len(text.split()) * 1.3
