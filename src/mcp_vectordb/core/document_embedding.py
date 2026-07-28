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

import logging
from typing import List, NamedTuple, Optional

from sympy import false, true
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)

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
    heading_path: str
    html: Optional[str] = None




class MultimodalExtractionResult(NamedTuple):
    """The full set of elements extracted from one document, split by category."""

    text_elements: List[ExtractedElement]
    table_elements: List[ExtractedElement]
    image_elements: List[ExtractedElement]


def extract_multimodal_document(file_path: str) -> MultimodalExtractionResult:
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
            infer_table_structure=True, # to get the table as html
            strategy="hi_res", # to use the layout detection
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
    active_headings = {}

    """
    for element in elements:
        page_number = getattr(element.metadata, "page_number", None)
        element_type = type(element).__name__


        current_path = ""
        if element_type == "Title" and element.text.strip():
            # Get the depth (default to 1 if unstructured couldn't figure it out)
            depth = getattr(element.metadata, "category_depth", 1)
            if depth is None:
                depth = 1

            # Save this heading at its specific depth level
            active_headings[depth] = element.text.strip()

            keys_to_remove = [k for k in active_headings.keys() if k > depth]
            for k in keys_to_remove:
                del active_headings[k]

            current_path = " > ".join(
                [active_headings[k] for k in sorted(active_headings.keys())]
            )

        if element.category == _TABLE_CATEGORY:
            html_content = getattr(element.metadata, "text_as_html", None)
            raw_text = getattr(element, "text", None)
            table_content: str = html_content or raw_text or ""

            table_elements.append(
                ExtractedElement(
                    category="table",
                    element_type=element_type,
                    page_number=page_number,
                    content=table_content,
                    html=getattr(element.metadata, "text_as_html", None),
                    heading_path= current_path,
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
                    heading_path= current_path,
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
                    heading_path= current_path,
                )
            )

    logger.info(
        "Extracted document '%s': %d text, %d table, %d image elements",
        file_path,
        len(text_elements),
        len(table_elements),
        len(image_elements),
    )
    """

    elementsChunks = chunk_by_title(elements,
                                    max_characters= 5000, # maximum character for the chunk
                                    new_after_n_chars= 1000,
                                    combine_text_under_n_chars=0 , # to disable merge of different heading
                                    isolate_table= false)

    for i, chunk in enumerate(elementsChunks):
        if i in (0,1):
            print(f"THE CHUNK IS{i}")
            for orig_el in chunk.metadata.orig_elements:
                print(orig_el.text)

                # Check for Table HTML
                if orig_el.category == "Table" and orig_el.metadata.text_as_html:
                    print("--- Found Table HTML ---")
                    print(orig_el.metadata.text_as_html)

                # Check for Base64 Encoded Image Data
                if orig_el.category == _IMAGE_CATEGORY:
                    print("--- Found Base64 Image Payload ---")
                    print(f"MIME Type: {orig_el.metadata.image_mime_type}")
                    print(f"Base64 String: {orig_el.metadata.image_base64}")




    return MultimodalExtractionResult(
        text_elements=text_elements,
        table_elements=table_elements,
        image_elements=image_elements,
    )


def _get_extension(file_path: str) -> str:
    import os

    _, extension = os.path.splitext(file_path)
    return extension.lower()
