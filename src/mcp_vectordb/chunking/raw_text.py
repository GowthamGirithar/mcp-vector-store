"""Raw text extraction for the agentic-embedding tool.

Unlike `process_document.py` (which chunks by title for the fixed-size
pipeline), the primary path here (`extract_raw_text`) extracts a document's
full plain text with no chunking — chunk boundaries are decided later by the
LLM in `core/agentic_chunking.py`. `extract_title_sections` is only used as
a fallback for documents too large to send to the LLM in a single call; see
`core/agentic_chunking.agentic_chunk_document`.
"""

import logging
from typing import List

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

from .process_document import (
    MULTIMODAL_SUPPORTED_EXTENSIONS,
    UnsupportedDocumentTypeError,
    DocumentEmbeddingParseError,
    _get_extension,
)

logger = logging.getLogger(__name__)


def _partition_document(file_path: str) -> List:
    extension = _get_extension(file_path)
    if extension not in MULTIMODAL_SUPPORTED_EXTENSIONS:
        allowed_str = ", ".join(sorted(MULTIMODAL_SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentTypeError(
            f"Unsupported file extension '{extension}' for file '{file_path}'. "
            f"Supported extensions are: {allowed_str}"
        )

    try:
        return partition(filename=file_path)
    except Exception as exc:
        raise DocumentEmbeddingParseError(
            f"Failed to parse document '{file_path}': {exc}"
        ) from exc


def extract_raw_text(file_path: str) -> str:
    """Extract a document's full plain text via `unstructured`, with no chunking.

    Args:
        file_path: Path to the document to extract. Must have an extension
            in `MULTIMODAL_SUPPORTED_EXTENSIONS`.

    Returns:
        The document's plain text, with elements joined by newlines.

    Raises:
        UnsupportedDocumentTypeError: If the file extension is not supported.
        DocumentEmbeddingParseError: If the file cannot be parsed.
    """
    elements = _partition_document(file_path)
    return "\n".join(element.text for element in elements if element.text)


def extract_title_sections(file_path: str, max_characters: int) -> List[str]:
    """Split a document into title-delimited sections of up to `max_characters`.

    Fallback for documents whose full raw text is too large for a single LLM
    call: splits on natural heading boundaries (via `unstructured`'s
    title-based chunker) instead of an arbitrary character cut, so each
    section stays coherent. Note that tasks needing to reason across
    sections (e.g. matching a question to an answer key in a different
    section) aren't guaranteed correct once a document is split this way,
    since each section is processed independently.

    Args:
        file_path: Path to the document to extract.
        max_characters: Target max size per section.

    Returns:
        The document's sections, in order, as plain text.

    Raises:
        UnsupportedDocumentTypeError: If the file extension is not supported.
        DocumentEmbeddingParseError: If the file cannot be parsed.
    """
    elements = _partition_document(file_path)
    chunks = chunk_by_title(
        elements,
        max_characters=max_characters,
        new_after_n_chars=max_characters,
        combine_text_under_n_chars=0,
        isolate_table=False,
    )
    return [chunk.text for chunk in chunks if chunk.text and chunk.text.strip()]
