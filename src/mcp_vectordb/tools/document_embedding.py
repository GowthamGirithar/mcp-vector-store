"""Multimodal document extraction tool for the MCP Vector Database Server.

Implements `generate_document_embedding`: parses a document with
`unstructured` into text/table/image elements (see
`core.document_embedding`). This tool performs extraction only — no
embedding generation and no vector DB storage (see
docs/generate-document-embedding/spec.md).
"""

import logging
import os

from mcp.server.fastmcp import Context

from ..server import mcp
from ..config.config import get_settings
from ..core.document_embedding import (
    DocumentEmbeddingParseError,
    MULTIMODAL_SUPPORTED_EXTENSIONS,
    extract_multimodal_document,
)
from ..utils.validation import validate_file_path
from ..utils.exceptions import ValidationError

logger = logging.getLogger(__name__)


@mcp.tool()
async def generate_document_embedding(file_path: str, ctx: Context = None) -> str:
    """Extract text, table, and image elements from a document using `unstructured`.

    This tool performs extraction only: it does not generate embeddings or
    store anything in the vector database.

    Args:
        file_path: Path to the document to extract (.pdf, .md, .docx, .pptx)
        ctx: FastMCP context for logging and progress reporting

    Returns:
        Success message with counts of extracted text/table/image chunks
    """
    try:
        settings = get_settings()

        validated_file_path = validate_file_path(
            file_path,
            allowed_extensions=MULTIMODAL_SUPPORTED_EXTENSIONS,
            max_file_size_mb=settings.document.max_file_size_mb,
        )
        source_filename = os.path.basename(validated_file_path)

        if ctx:
            await ctx.info(f"Starting document extraction: {validated_file_path}")

        result = extract_multimodal_document(validated_file_path)

        for element in result.text_elements + result.table_elements + result.image_elements:
            logger.debug(
                f"Extracted {element.category} element "
                f"(type={element.element_type}, page={element.page_number}): "
                f"{element.content!r}"
            )

        if ctx:
            await ctx.info(
                f"Successfully extracted document '{source_filename}': "
                f"{len(result.text_elements)} text, {len(result.table_elements)} table, "
                f"{len(result.image_elements)} image chunk(s)"
            )

        return (
            f"Successfully extracted document '{source_filename}'\n"
            f"Text chunks: {len(result.text_elements)}\n"
            f"Table chunks: {len(result.table_elements)}\n"
            f"Image chunks: {len(result.image_elements)}"
        )

    except ValidationError as e:
        error_msg = f"Validation error: {e.message}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)

    except DocumentEmbeddingParseError as e:
        error_msg = f"Document parse error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)

    except ValueError as e:
        # Catches UnsupportedDocumentTypeError (a ValueError subclass) and any
        # other plain ValueError raised below this point.
        error_msg = f"Validation error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise ValueError(error_msg)

    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        raise RuntimeError(error_msg)
