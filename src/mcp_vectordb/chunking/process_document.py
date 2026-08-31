"""Multimodal document extraction for the generate-document-embedding tool.

Provides `process_document`, which partitions a document into chunks and,
for each chunk, collects:

- text: the chunk's plain text.
- tableHTML: the HTML representation of every table found under that chunk
  — a chunk may contain more than one.
- imageBase64: the base64-encoded bytes of every image found under that
  chunk — a chunk may contain more than one.

Two interchangeable parsers are supported via the `parser` argument
(default `"unstructured"`):

- `"unstructured"`: partitions with the `unstructured` library, then splits
  into title-delimited chunks via `chunk_by_title`. Known limitation:
  `unstructured` only extracts embedded images out of the box for PDF.
  DOCX/PPTX image extraction is a pluggable extension point with no built-in
  implementation, so `imageBase64` is always empty for those formats (see
  docs/generate-document-embedding/spec.md).
- `"docling"`: converts with Docling's `DocumentConverter`, then chunks via
  `HybridChunker`. Known limitation: `HybridChunker` only emits `doc_items`
  that carry text, so a caption-less picture is never attributed to any
  chunk (even though Docling detects and can render it) — `imageBase64`
  will be empty in that case, regardless of format.

This module implements extraction only — no embedding generation, no
further chunking, and no vector DB storage (see docs/generate-document-embedding/).
"""

import base64
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from io import BytesIO
from typing import Callable, Dict, List, Literal, NamedTuple, Optional

import fitz  # PyMuPDF
from docling_core.transforms.chunker import HybridChunker

from unstructured.chunking.title import chunk_by_title
from unstructured.partition.docx import partition_docx
from unstructured.partition.md import partition_md
from unstructured.partition.pdf import partition_pdf
from unstructured.partition.pptx import partition_pptx

from .table_text import linearize_table_html
from .token_budget import split_text_to_budget

logger = logging.getLogger(__name__)

# Chunks under this many tokens are merged into their preceding sibling
# (within the same chunk_by_title group) rather than embedded on their own —
# see `token_budget.fit_chunks_to_budget`.
DEFAULT_MIN_CHUNK_TOKENS = 20
# Default token overlap carried between pieces of a chunk split for being
# over the embedding model's budget.
DEFAULT_CHUNK_OVERLAP_TOKENS = 32

MULTIMODAL_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".docx", ".pptx"}
SUPPORTED_PARSERS = ("unstructured", "docling")

_TABLE_CATEGORY = "Table"
_IMAGE_CATEGORY = "Image"


class UnsupportedDocumentTypeError(ValueError):
    """Raised when `process_document` is given a file with an unsupported extension."""


class DocumentEmbeddingParseError(Exception):
    """Raised when a document of a supported type cannot be parsed by the selected parser."""


class MultimodalExtractionResult(NamedTuple):
    """The elements extracted from one title-delimited chunk of a document."""

    text: str
    tableHTML: List[str]
    imageBase64: List[str]

def _process_pdf_range(file_path: str, start_page: int, end_page: int) -> List:
    """Worker function to process a range of PDF pages with `hi_res`.

    Every PDF page is partitioned with `hi_res` (layout-model-based
    extraction) rather than being classified into `hi_res`/`fast` per page:
    the prior classifier (`_is_complex_pdf_page`) always returned `True`
    regardless of page content, so every page was already going through
    `hi_res` in practice — this makes that the explicit, intended behavior
    instead of a dead classification path, and accepts the cost of running
    the layout model on plain-text pages in exchange for correctly capturing
    tables (`infer_table_structure=True`) and images on every page,
    including ones that a lighter classifier might misjudge as simple.
    `partition_pdf` is still called once per assigned page range (not once
    per page) so the layout model loads once per range rather than once per
    page.
    """
    try:
        return partition_pdf(
            filename=file_path,
            strategy="hi_res",
            starting_page_number=start_page,
            ending_page_number=end_page,
            infer_table_structure=True,
        )
    except Exception as e:
        logger.error(
            f"Error processing pages {start_page}-{end_page} of '{file_path}': {e}"
        )
        return []


def _process_docling(file_path: str) -> List[MultimodalExtractionResult]:
    """Partition `file_path` with `docling`, chunking via `HybridChunker`."""
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.chunking import HybridChunker

    # 1. Configure converter to generate picture images and decode formulas.
    # Docling detects equations as `TextItem`s with `label="formula"` but
    # leaves `.text` empty unless formula enrichment is turned on — without
    # this, every formula contributes nothing to any chunk's text.
    pdf_pipeline_options = PdfPipelineOptions()
    pdf_pipeline_options.generate_picture_images = True
    pdf_pipeline_options.do_formula_enrichment = True

    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options)}
    )
    doc = converter.convert(file_path).document

    # 2. Use HierarchicalChunker to generate structure-aware chunks
    chunker = HybridChunker()

    # Fast lookup for tables and pictures - it has only reference
    tables_by_ref = {table.self_ref: table for table in doc.tables}
    pictures_by_ref = {pic.self_ref: pic for pic in doc.pictures}
    attributed_picture_refs: set = set()

    chunks = list(chunker.chunk(doc))

    texts: List[str] = []
    table_htmls: List[List[str]] = []
    image_b64s: List[List[str]] = []
    # doc_items' self_refs per chunk, for exact caption lookup
    chunk_item_refs: List[set] = []

    for chunk in chunks:
        table_html: List[str] = []
        image_base64: List[str] = []
        item_refs: set = set()

        for item in chunk.meta.doc_items:
            item_refs.add(item.self_ref)

            # Extract Tables under this chunk/heading
            table = tables_by_ref.get(item.self_ref)
            if table is not None:
                table_html.append(table.export_to_html(doc=doc))

            # Extract Pictures under this chunk/heading
            picture = pictures_by_ref.get(item.self_ref)
            if picture is not None:
                encoded = _encode_picture(picture, doc)
                if encoded is not None:
                    image_base64.append(encoded)
                    attributed_picture_refs.add(picture.self_ref)

        texts.append(chunker.contextualize(chunk))
        table_htmls.append(table_html)
        image_b64s.append(image_base64)
        chunk_item_refs.append(item_refs)

    # Fallback pass: HybridChunker only attaches doc_items that carry text, so a
    # picture is never a chunk's doc_item on its own. If the picture has a
    # caption, attach it to the chunk containing that caption's text item
    # (exact self_ref match). Caption-less pictures are intentionally left
    # unattached rather than guessed at via page/position — see the module
    # docstring's known-limitation note.
    for picture in doc.pictures:
        if picture.self_ref in attributed_picture_refs:
            continue

        target_idx = _find_chunk_by_caption(picture, doc, chunk_item_refs)
        if target_idx is None:
            continue

        encoded = _encode_picture(picture, doc)
        if encoded is not None:
            image_b64s[target_idx].append(encoded)

    return [
        MultimodalExtractionResult(text=text, tableHTML=table_html, imageBase64=image_base64)
        for text, table_html, image_base64 in zip(texts, table_htmls, image_b64s)
    ]


def _find_chunk_by_caption(picture, doc, chunk_item_refs: List[set]):
    """Return the index of the chunk containing `picture`'s caption text item, if any."""
    for caption_ref in picture.captions:
        caption_self_ref = caption_ref.resolve(doc).self_ref
        for i, item_refs in enumerate(chunk_item_refs):
            if caption_self_ref in item_refs:
                return i
    return None


def _encode_picture(picture, doc) -> str | None:
    pil_image = picture.get_image(doc)
    if pil_image is None:
        return None
    buffer = BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def process_document(
    file_path: str,
    batch_size: int = 25,
    max_workers: int = 4,
    parser: Literal["unstructured", "docling"] = "unstructured",
    count_tokens: Optional[Callable[[str], int]] = None,
    max_tokens: Optional[int] = None,
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS,
) -> List[MultimodalExtractionResult]:
    """Partition `file_path` into multimodal chunks using the selected `parser`.

    `parser="unstructured"` (default) runs the existing parallel hybrid
    pipeline. `parser="docling"` runs Docling's `DocumentConverter` +
    `HybridChunker` instead, for side-by-side comparison — it ignores
    `batch_size`/`max_workers`, which are `unstructured`-specific.

    `count_tokens`/`max_tokens` (both required together to take effect —
    pass the embedding service's own `count_tokens`/`max_input_tokens`) size
    each chunk's text to the target embedding model's token budget: any
    chunk that would otherwise be truncated by the model is split on
    sentence boundaries into pieces that each fit, with `chunk_overlap_tokens`
    of overlap carried between adjacent pieces so a fact sitting on a
    sentence boundary isn't stranded in only one piece. Table/image payloads
    ride with the first piece of the chunk they came from. Omitting both
    (the default) skips this pass entirely — the `docling` parser is
    unaffected either way, since `HybridChunker` already tokenizes internally.
    """
    extension = _get_extension(file_path)
    if extension not in MULTIMODAL_SUPPORTED_EXTENSIONS:
        allowed_str = ", ".join(sorted(MULTIMODAL_SUPPORTED_EXTENSIONS))
        raise UnsupportedDocumentTypeError(
            f"Unsupported file extension '{extension}' for file '{file_path}'. "
            f"Supported extensions are: {allowed_str}"
        )

    if parser not in SUPPORTED_PARSERS:
        raise ValueError(
            f"Unknown parser '{parser}'. Supported parsers are: {', '.join(SUPPORTED_PARSERS)}"
        )

    if parser == "docling":
        try:
            return _process_docling(file_path)
        except Exception as exc:
            raise DocumentEmbeddingParseError(
                f"Failed to parse document '{file_path}' with docling: {exc}"
            ) from exc

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
        # 200 chars keeps chunk_by_title's own title-boundary-respecting merge
        # from leaving slivers behind — measured on attention.pdf: combine=0
        # left 54% of chunks under 10 tokens (median 7); combine=200 clears
        # every sub-10-token chunk while barely moving the oversize count.
        combine_text_under_n_chars=200,
        isolate_table=False,
    )

    results: List[MultimodalExtractionResult] = []
    for chunk in chunks:
        text_parts: List[str] = []
        table_html: List[str] = []
        image_base64: List[str] = []

        for orig_el in chunk.metadata.orig_elements or []:
            if orig_el.category == _TABLE_CATEGORY:
                html = orig_el.metadata.text_as_html
                table_html.append(html)
                # Table content used to be dropped from the embedded/searched
                # text entirely (only `has_table: True` survived to metadata)
                # — linearize it into the chunk's own text in document order,
                # right where the table appeared, so its content is
                # retrievable via both vector and BM25 search.
                linearized = linearize_table_html(html)
                if linearized:
                    text_parts.append(linearized)
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

    if count_tokens is not None and max_tokens is not None:
        results = _split_oversize_results(results, count_tokens, max_tokens, chunk_overlap_tokens)

    return results


def _split_oversize_results(
    results: List[MultimodalExtractionResult],
    count_tokens: Callable[[str], int],
    max_tokens: int,
    overlap: int,
) -> List[MultimodalExtractionResult]:
    """Split any chunk whose text would be truncated by the embedding model.

    Only splits — fragmentation is already handled upstream by
    `combine_text_under_n_chars` in the `chunk_by_title` call above, which
    (unlike a post-hoc merge here) can merge slivers without crossing the
    title boundaries `chunk_by_title` itself decided on.
    """
    split_results: List[MultimodalExtractionResult] = []
    for result in results:
        pieces = split_text_to_budget(result.text, count_tokens, max_tokens, overlap)
        if len(pieces) <= 1:
            split_results.append(result)
            continue
        for i, piece in enumerate(pieces):
            # table/image payloads ride with the first piece only, so a
            # table is never duplicated across a chunk's split pieces
            split_results.append(
                MultimodalExtractionResult(
                    text=piece,
                    tableHTML=result.tableHTML if i == 0 else [],
                    imageBase64=result.imageBase64 if i == 0 else [],
                )
            )
    return split_results


def _get_extension(file_path: str) -> str:
    _, extension = os.path.splitext(file_path)
    return extension.lower()
