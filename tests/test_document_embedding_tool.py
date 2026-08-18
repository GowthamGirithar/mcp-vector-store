"""Tests for the generate_document_embedding MCP tool."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_vectordb.chunking.process_document import process_document
from mcp_vectordb.config.config import DocumentConfig
from mcp_vectordb.tools import document_embedding as document_embedding_tool

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


CASES = {
    "multimodal_sample.pdf": {"table_chunks": 1, "image_chunks": 1},
    "multimodal_sample.docx": {"table_chunks": 1, "image_chunks": 0},
    "multimodal_sample.pptx": {"table_chunks": 1, "image_chunks": 0},
    "multimodal_sample.md": {"table_chunks": 1, "image_chunks": 0},
}


@pytest.mark.asyncio
@pytest.mark.parametrize("filename,expected", CASES.items(), ids=CASES.keys())
async def test_generate_document_embedding_reports_chunk_counts(filename, expected, real_services):
    result = await document_embedding_tool.generate_document_embedding(
        file_path=fixture_path(filename)
    )

    assert filename in result
    assert "Text chunks: " in result
    assert f"Table chunks: {expected['table_chunks']}" in result
    assert f"Image chunks: {expected['image_chunks']}" in result

def test_process_document_docling_attaches_tables_and_images():
    """Regression test: `_process_docling` used to crash immediately because
    `HybridChunker` was never instantiated (`chunker = HybridChunker`).
    `HybridChunker` only attaches doc_items that carry text, so a picture is
    never a chunk's doc_item on its own; captioned pictures are attached to
    the chunk containing their caption text, caption-less pictures are left
    unattached (not guessed at via page/position)."""
    results = process_document(fixture_path("attention.pdf"), parser="docling")

    assert len(results) > 0
    assert any(chunk.tableHTML for chunk in results)
    assert any(chunk.imageBase64 for chunk in results)


def test_process_document_docling_decodes_formulas():
    """Regression test: Docling detects equations as label="formula" text
    items but leaves `.text` empty unless `do_formula_enrichment` is enabled,
    so a formula previously contributed nothing to any chunk's text."""
    results = process_document(fixture_path("attention.pdf"), parser="docling")

    assert any("softmax" in chunk.text and "frac" in chunk.text for chunk in results)


@pytest.mark.asyncio
async def test_generate_document_embedding_uses_configured_parser():
    """The tool must read the parser choice from settings.document.parser
    rather than hardcoding a specific parser.

    Vector DB / embedding service are mocked here so this test isolates the
    parser-selection behavior from the real service wiring (which requires
    the FastMCP server lifespan to have run to populate those globals)."""
    settings = document_embedding_tool.get_settings()
    settings.document.parser = "docling"

    mock_vector_db = MagicMock()
    mock_vector_db.collection_exists = AsyncMock(return_value=True)
    mock_vector_db.store_documents = AsyncMock(return_value=["chunk-id"])

    mock_embedding_service = MagicMock()
    mock_embedding_service.dimension = 384
    mock_embedding_service.generate_embeddings = AsyncMock(return_value=[[0.0] * 384])

    with patch.object(
        document_embedding_tool, "process_document", wraps=process_document
    ) as mock_process_document, patch.object(
        document_embedding_tool, "get_vector_db", return_value=mock_vector_db
    ), patch.object(
        document_embedding_tool, "get_embedding_service", return_value=mock_embedding_service
    ):
        await document_embedding_tool.generate_document_embedding(
            file_path=fixture_path("multimodal_sample.md")
        )
        mock_process_document.assert_called_once()
        assert mock_process_document.call_args.kwargs["parser"] == "docling"


def test_document_config_rejects_unsupported_parser():
    with pytest.raises(ValueError):
        DocumentConfig(parser="not-a-real-parser")
