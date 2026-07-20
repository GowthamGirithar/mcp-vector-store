"""Text extraction for uploaded documents.

Provides `extract_text`, which dispatches on a file's lowercased extension
and returns a list of `(page_number, text)` pairs:

- `.pdf`: one tuple per page, `page_number` starting at 1.
- `.txt` / `.md`: a single tuple `(None, text)` with the file's full content.
- any other extension: raises `UnsupportedFileTypeError` before the file is
  opened or parsed.

A corrupt or otherwise unparseable PDF raises `DocumentParseError` (wrapping
the underlying `pypdf` exception) so callers can distinguish "bad input"
from unrelated bugs and map it cleanly (e.g. to a `RuntimeError` at the tool
layer).
"""

from typing import List, Optional, Tuple

from pypdf import PdfReader
from pypdf.errors import PdfReadError


class UnsupportedFileTypeError(ValueError):
    """Raised when `extract_text` is given a file with an unsupported extension."""


class DocumentParseError(Exception):
    """Raised when a document of a supported type cannot be parsed (e.g. a corrupt PDF)."""


_TEXT_EXTENSIONS = {".txt", ".md"}


def extract_text(file_path: str) -> List[Tuple[Optional[int], str]]:
    """Extract text from a document, returning `(page_number, text)` pairs.

    Args:
        file_path: Path to the document to extract text from.

    Returns:
        A list of `(page_number, text)` tuples. For PDFs, one tuple per page
        with `page_number` starting at 1. For `.txt`/`.md` files, a single
        tuple `(None, text)`.

    Raises:
        UnsupportedFileTypeError: If the file extension is not one of
            `.pdf`, `.txt`, `.md`. Raised before the file is opened.
        DocumentParseError: If a `.pdf` file cannot be parsed (e.g. it is
            corrupt or truncated).
    """
    extension = _get_lowercased_extension(file_path)

    if extension == ".pdf":
        return _extract_pdf_text(file_path)
    if extension in _TEXT_EXTENSIONS:
        return _extract_plain_text(file_path)

    raise UnsupportedFileTypeError(
        f"Unsupported file extension '{extension}' for file '{file_path}'. "
        f"Supported extensions are: .pdf, .txt, .md"
    )


def _get_lowercased_extension(file_path: str) -> str:
    _, dot_extension = _splitext(file_path)
    return dot_extension.lower()


def _splitext(file_path: str) -> Tuple[str, str]:
    import os

    return os.path.splitext(file_path)


def _extract_plain_text(file_path: str) -> List[Tuple[Optional[int], str]]:
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()
    return [(None, text)]


def _extract_pdf_text(file_path: str) -> List[Tuple[Optional[int], str]]:
    try:
        reader = PdfReader(file_path)
        pages = [
            (page_number, page.extract_text())
            for page_number, page in enumerate(reader.pages, start=1)
        ]
    except PdfReadError as exc:
        raise DocumentParseError(
            f"Failed to parse PDF '{file_path}': {exc}"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive: any other pypdf failure
        raise DocumentParseError(
            f"Failed to parse PDF '{file_path}': {exc}"
        ) from exc

    return pages
