"""Multimodal document extraction for the generate-document-embedding tool.

Provides `extract_multimodal`, which partitions a document with the
`unstructured` library and separates its elements into text, table, and
image categories:

- text: plain text content (category-agnostic — everything that isn't a
  table or an image, e.g. `Title`, `NarrativeText`, `ListItem`).
- table: HTML representation (via `infer_table_structure`), when available,
  alongside the table's plain-text content.
- image: base64-encoded bytes (via `extract_image_block_to_payload`, no
  disk writes).

Known limitation: `unstructured` only extracts embedded images out of the
box for PDF. DOCX/PPTX image extraction is a pluggable extension point with
no built-in implementation, so `image_elements` is always empty for those
formats (see docs/generate-document-embedding/spec.md).

This module implements extraction only — no embedding generation, no
chunking, and no vector DB storage (see docs/generate-document-embedding/).
"""

from typing import List, NamedTuple, Optional

from unstructured.partition.auto import partition

MULTIMODAL_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".docx", ".pptx"}

_TABLE_CATEGORY = "Table"
_IMAGE_CATEGORY = "Image"


class UnsupportedDocumentTypeError(ValueError):
    """Raised when `extract_multimodal` is given a file with an unsupported extension."""


class DocumentEmbeddingParseError(Exception):
    """Raised when a document of a supported type cannot be parsed by `unstructured`."""


class ExtractedElement(NamedTuple):
    """One element produced by `extract_multimodal`.

    ``content`` is the plain text for text elements, the table's plain-text
    content for table elements, and base64-encoded image bytes for image
    elements. ``html`` is only populated for table elements (when
    `unstructured` was able to infer table structure).
    """

    category: str
    element_type: str
    page_number: Optional[int]
    content: str
    html: Optional[str] = None


class MultimodalExtractionResult(NamedTuple):
    """The full set of elements extracted from one document, split by category."""

    text_elements: List[ExtractedElement]
    table_elements: List[ExtractedElement]
    image_elements: List[ExtractedElement]


def extract_multimodal(file_path: str) -> MultimodalExtractionResult:
    """Partition `file_path` with `unstructured` into text/table/image elements.

    Args:
        file_path: Path to the document to extract. Must have an extension
            in `MULTIMODAL_SUPPORTED_EXTENSIONS`.

    Returns:
        A `MultimodalExtractionResult` with the document's elements split
        into text, table, and image categories.

    Raises:
        UnsupportedDocumentTypeError: If the file extension is not in
            `MULTIMODAL_SUPPORTED_EXTENSIONS`. Raised before the file is
            opened.
        DocumentEmbeddingParseError: If the file cannot be parsed (e.g. it
            is corrupt or truncated).
    """
    extension = _get_extension(file_path)
    if extension not in MULTIMODAL_SUPPORTED_EXTENSIONS:
        allowed_str = ", ".join(sorted(MULTIMODAL_SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentTypeError(
            f"Unsupported file extension '{extension}' for file '{file_path}'. "
            f"Supported extensions are: {allowed_str}"
        )

    try:
        elements = partition(
            filename=file_path,
            infer_table_structure=True,
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )
    except Exception as exc:
        raise DocumentEmbeddingParseError(
            f"Failed to parse document '{file_path}': {exc}"
        ) from exc

    text_elements: List[ExtractedElement] = []
    table_elements: List[ExtractedElement] = []
    image_elements: List[ExtractedElement] = []

    for element in elements:
        page_number = getattr(element.metadata, "page_number", None)
        element_type = type(element).__name__

        if element.category == _TABLE_CATEGORY:
            table_elements.append(
                ExtractedElement(
                    category="table",
                    element_type=element_type,
                    page_number=page_number,
                    content=element.text,
                    html=getattr(element.metadata, "text_as_html", None),
                )
            )
        elif element.category == _IMAGE_CATEGORY:
            image_base64 = getattr(element.metadata, "image_base64", None)
            if image_base64 is None:
                continue
            image_elements.append(
                ExtractedElement(
                    category="image",
                    element_type=element_type,
                    page_number=page_number,
                    content=image_base64,
                )
            )
        else:
            if not element.text or not element.text.strip():
                continue
            text_elements.append(
                ExtractedElement(
                    category="text",
                    element_type=element_type,
                    page_number=page_number,
                    content=element.text,
                )
            )

    return MultimodalExtractionResult(
        text_elements=text_elements,
        table_elements=table_elements,
        image_elements=image_elements,
    )


def _get_extension(file_path: str) -> str:
    import os

    _, extension = os.path.splitext(file_path)
    return extension.lower()
