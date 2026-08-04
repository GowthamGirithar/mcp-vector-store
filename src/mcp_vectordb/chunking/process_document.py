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
from concurrent.futures import ProcessPoolExecutor
from typing import Callable, Dict, List, NamedTuple

import fitz  # PyMuPDF

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.docx import partition_docx
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx

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

def _is_complex_pdf_page(page: fitz.Page) -> bool:
    if len(page.get_images()) > 0:
        return True

    if len(page.get_drawings()) >0:
        return True

    return True


def _process_pdf_range(file_path: str, start_page: int, end_page: int) -> List:
    """Worker function to process a range of PDF pages using Hybrid strategy.

    Pages are grouped into contiguous runs by strategy so `partition_pdf` (and
    its hi_res layout-model load) is called once per run instead of once per
    page — calling it per page was re-initializing the layout model on every
    page, dominating runtime even for small documents.
    """
    doc = fitz.open(file_path)
    strategies = [
        "hi_res" if _is_complex_pdf_page(doc[page_idx - 1]) else "fast"
        for page_idx in range(start_page, end_page + 1)
    ]
    doc.close()

    runs = []
    run_start = start_page
    for offset in range(1, len(strategies)):
        if strategies[offset] != strategies[offset - 1]:
            runs.append((run_start, start_page + offset - 1, strategies[offset - 1]))
            run_start = start_page + offset
    runs.append((run_start, end_page, strategies[-1]))

    range_elements = []
    for run_start_page, run_end_page, strategy in runs:
        try:
            logger.info(
                "Strategy and page range %s %s-%s", strategy, run_start_page, run_end_page
            )

            elements = partition_pdf(
                filename=file_path,
                strategy=strategy,
                starting_page_number=run_start_page,
                ending_page_number=run_end_page,
                infer_table_structure=(strategy == "hi_res"),
            )
            range_elements.extend(elements)
        except Exception as e:
            logger.error(
                f"Error processing pages {run_start_page}-{run_end_page} of '{file_path}': {e}"
            )

    return range_elements


def process_document(file_path: str, batch_size:int = 25, max_workers: int = 4) -> List[MultimodalExtractionResult]:
    """Partition `file_path` with `unstructured` using parallel hybrid execution."""
    extension = _get_extension(file_path)
    if extension not in MULTIMODAL_SUPPORTED_EXTENSIONS:
        allowed_str = ", ".join(sorted(MULTIMODAL_SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentTypeError(
            f"Unsupported file extension '{extension}' for file '{file_path}'. "
            f"Supported extensions are: {allowed_str}"
        )

    elements = []
    try:
        if extension == ".pdf":
            doc = fitz.open(file_path)
            total_pages = len(doc)
            doc.close()

            logger.info(
                f"Processing PDF '{file_path}' ({total_pages} pages) across {max_workers} processes..."
            )

            if total_pages < 30:
                logger.info(f"Small PDF detected ({total_pages} pages). Running sequentially...")
                elements = _process_pdf_range(file_path, 1, total_pages)
            else:
                # Build 1-indexed page ranges
                ranges = [
                    (i, min(i + batch_size - 1, total_pages))
                    for i in range(1, total_pages + 1, batch_size)
                ]

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    futures = [
                        executor.submit(_process_pdf_range, file_path, start, end)
                        for start, end in ranges
                    ]
                    for future in futures:
                        elements.extend(future.result())

        else:
            handlers: Dict[str, Callable] = {
                ".docx": lambda path: partition_docx(filename=path, infer_table_structure=True),
                ".pptx": lambda path: partition_pptx(filename=path, infer_table_structure=True, include_slide_notes=True),
                ".md": lambda path: partition_md(filename=path),
            }
            elements = handlers[extension](file_path)

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
