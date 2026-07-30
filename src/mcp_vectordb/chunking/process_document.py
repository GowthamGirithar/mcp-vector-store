"""Multimodal document extraction for the generate-document-embedding tool.

Provides `process_document`, which partitions a document with the
`unstructured` library, splits it into title-delimited chunks, and for each
chunk collects:

- text: the concatenated plain text of the chunk's non-table/non-image
  elements (e.g. `Title`, `NarrativeText`, `ListItem`).
- tableHTML: the HTML representation (via `infer_table_structure`) of every
  table found under that chunk's title — a chunk may contain more than one.
- imageBase64: the base64-encoded bytes (via `extract_image_block_to_payload`,
  no disk writes) of every image found under that chunk's title — a chunk
  may contain more than one.

Known limitation: `unstructured` only extracts embedded images out of the
box for PDF. DOCX/PPTX image extraction is a pluggable extension point with
no built-in implementation, so `imageBase64` is always empty for those
formats (see docs/generate-document-embedding/spec.md).

This module implements extraction only — no embedding generation, no
further chunking, and no vector DB storage (see docs/generate-document-embedding/).
"""

import logging
import os
from typing import List, NamedTuple

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition

logger = logging.getLogger(__name__)

MULTIMODAL_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".docx", ".pptx"}

_TABLE_CATEGORY = "Table"
_IMAGE_CATEGORY = "Image"


class UnsupportedDocumentTypeError(ValueError):
    """Raised when `process_document` is given a file with an unsupported extension."""


class DocumentEmbeddingParseError(Exception):
    """Raised when a document of a supported type cannot be parsed by `unstructured`."""


class MultimodalExtractionResult(NamedTuple):
    """The elements extracted from one title-delimited chunk of a document."""

    text: str
    tableHTML: List[str]
    imageBase64: List[str]


def process_document(file_path: str) -> List[MultimodalExtractionResult]:
    """Partition `file_path` with `unstructured` and split it into
    title-delimited chunks, one `MultimodalExtractionResult` per chunk.

    Args:
        file_path: Path to the document to extract. Must have an extension
            in `MULTIMODAL_SUPPORTED_EXTENSIONS`.

    Returns:
        One `MultimodalExtractionResult` per title-delimited chunk, each
        holding that chunk's text and every table/image found under it.

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
            infer_table_structure=True,  # to get the table as html
            strategy="hi_res",  # to use the layout detection
            extract_image_block_types=["Image"],
            extract_image_block_to_payload=True,
        )
    except Exception as exc:
        raise DocumentEmbeddingParseError(
            f"Failed to parse document '{file_path}': {exc}"
        ) from exc

    chunks = chunk_by_title(
        elements,
        max_characters=5000,  # maximum characters per chunk
        new_after_n_chars=5000,
        combine_text_under_n_chars=0,  # disable merging across headings
        isolate_table=False,
    )

    results: List[MultimodalExtractionResult] = []
    for chunk in chunks:
        text_parts: List[str] = []
        table_html: List[str] = []
        image_base64: List[str] = []

        for orig_el in chunk.metadata.orig_elements or []:
            if orig_el.category == _TABLE_CATEGORY:
                table_html.append(orig_el.metadata.text_as_html)
            elif orig_el.category == _IMAGE_CATEGORY:
                image_base64.append(orig_el.metadata.image_base64)
            elif orig_el.text:
                text_parts.append(orig_el.text)

        results.append(
            MultimodalExtractionResult(
                text="\n".join(text_parts),
                tableHTML=table_html,
                imageBase64=image_base64,
            )
        )

    return results


def _get_extension(file_path: str) -> str:
    _, extension = os.path.splitext(file_path)
    return extension.lower()
