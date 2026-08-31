"""Linearize table HTML into retrievable plain text.

`unstructured` and `docling` both hand back tables as HTML. HTML is the right
shape for rendering but the wrong shape for retrieval: embedded as-is the tag
soup dominates the token budget, and stripped naively to
``"Name Score Alice 92 Bob 85"`` the association between a header and its
cell is gone — a query for *"Alice's score"* has no reason to prefer the row
that answers it.

`linearize_table_html` emits one line per body row with each cell prefixed by
its column header (``"Name: Alice | Score: 92"``), so the header and its value
sit adjacent in the embedded text and in the BM25 term positions.
"""

import logging
from html.parser import HTMLParser
from typing import List, Optional

logger = logging.getLogger(__name__)

_CELL_TAGS = ("td", "th")
_MAX_HEADER_CHARS = 80


class _TableParser(HTMLParser):
    """Collect a table's cell text as a list of rows, tracking header cells."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: List[List[str]] = []
        self.header_row_index: Optional[int] = None
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None
        self._row_is_header = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag == "tr":
            self._row = []
            self._row_is_header = False
        elif tag in _CELL_TAGS:
            if self._row is None:  # a cell outside any <tr>
                self._row = []
            self._cell = []
            if tag == "th":
                self._row_is_header = True

    def handle_endtag(self, tag: str) -> None:
        if tag in _CELL_TAGS and self._cell is not None:
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row is not None:
            self._close_row()

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def close(self) -> None:
        super().close()
        if self._cell is not None:  # unclosed final cell
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        if self._row:
            self._close_row()

    def _close_row(self) -> None:
        if any(cell for cell in self._row):
            if self._row_is_header and self.header_row_index is None:
                self.header_row_index = len(self.rows)
            self.rows.append(self._row)
        self._row = None


def linearize_table_html(html: str) -> str:
    """Render `html` as one ``"Header: cell | Header: cell"`` line per row.

    Falls back to pipe-joined cells for tables with no header row, and to the
    empty string for input that yields no cells at all. Never raises: a table
    that cannot be parsed contributes nothing to the chunk text rather than
    failing the whole document.
    """
    if not html:
        return ""

    parser = _TableParser()
    try:
        parser.feed(html)
        parser.close()
    except Exception as exc:  # malformed markup from an upstream parser
        logger.warning("Could not linearize table HTML (%s); skipping table text", exc)
        return ""

    rows = parser.rows
    if not rows:
        return ""

    header_index = parser.header_row_index
    if header_index is None and len(rows) > 1 and _looks_like_header(rows[0]):
        header_index = 0

    if header_index is None:
        return "\n".join(" | ".join(cell for cell in row if cell) for row in rows)

    headers = rows[header_index]
    lines = [" | ".join(cell for cell in headers if cell)]
    for row in rows[header_index + 1:]:
        parts = []
        for position, cell in enumerate(row):
            if not cell:
                continue
            header = headers[position] if position < len(headers) else ""
            parts.append(f"{header}: {cell}" if header else cell)
        if parts:
            lines.append(" | ".join(parts))
    return "\n".join(lines)


def _looks_like_header(row: List[str]) -> bool:
    """Treat a short, fully-populated, non-numeric first row as a header."""
    cells = [cell for cell in row if cell]
    if len(cells) != len(row) or not cells:
        return False
    return all(
        len(cell) <= _MAX_HEADER_CHARS and not cell.replace(".", "", 1).replace("-", "", 1).isdigit()
        for cell in cells
    )
