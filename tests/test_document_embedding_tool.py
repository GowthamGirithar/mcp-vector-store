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


def test_process_document_unstructured_embeds_table_content_in_text():
    """Regression test for the table-text-dropped gap: `process_document`
    used to route a Table element's content into `tableHTML` only, so
    `has_table: True` was the only trace of it left in the embedded/searched
    text — the actual cell values (e.g. "Alice", "92") were unsearchable.
    """
    results = process_document(fixture_path("multimodal_sample.docx"), parser="unstructured")

    assert any(r.tableHTML for r in results)
    table_chunk = next(r for r in results if r.tableHTML)
    assert "Alice" in table_chunk.text
    assert "92" in table_chunk.text


def test_process_document_splits_chunks_over_the_embedding_token_budget():
    """Regression test for the embedding-truncation gap: a chunk whose
    character count is under `chunk_by_title`'s 5000-char ceiling can still
    tokenize past the embedding model's `max_seq_length` (measured: 13 of
    157 chunks on attention.pdf exceeded MiniLM's 256-token limit, losing
    ~18% of the document's tokens). `count_tokens`/`max_tokens` must split
    those chunks so no piece exceeds the budget.
    """

    def count_tokens(text: str) -> int:
        return len(text.split())  # word count stands in for a real tokenizer

    long_text_chunks = process_document(
        fixture_path("attention.pdf"),
        parser="unstructured",
        count_tokens=count_tokens,
        max_tokens=50,
    )

    assert len(long_text_chunks) > 0
    for chunk in long_text_chunks:
        assert count_tokens(chunk.text) <= 50, (
            f"chunk exceeds token budget: {count_tokens(chunk.text)} > 50"
        )


@pytest.mark.asyncio
async def test_generate_embedding_warns_when_input_exceeds_model_token_limit(real_services, caplog):
    """Regression test: `SentenceTransformerEmbeddingService` used to embed
    (and silently truncate) any input regardless of length, with no signal
    that truncation had occurred."""
    _, embedding_service = real_services
    long_text = "word " * (embedding_service.max_input_tokens * 3)

    with caplog.at_level("WARNING"):
        await embedding_service.generate_embedding(long_text)

    assert any("truncated" in record.message.lower() for record in caplog.records)
