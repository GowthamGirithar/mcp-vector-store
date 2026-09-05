"""Integration tests for the generate_document_embedding MCP tool, exercised
against real (temp-directory) vector DB + embedding services via the
`real_services` fixture. See tests/unit/test_document_embedding_tool.py for
the mocked-service unit tests."""

import os

import pytest

from mcp_vectordb.tools import document_embedding as document_embedding_tool

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


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
    # Pin the parser explicitly rather than relying on whatever
    # DOCUMENT_PARSER the environment's .env resolves to (docling here) —
    # the expected counts below are specific to unstructured's extraction
    # behavior for these fixtures (e.g. docling only attaches a
    # caption-less image to no chunk at all, see
    # test_process_document_docling_attaches_tables_and_images).
    settings = document_embedding_tool.get_settings()
    settings.document.parser = "unstructured"

    result = await document_embedding_tool.generate_document_embedding(
        file_path=fixture_path(filename)
    )

    assert filename in result
    assert "Text chunks: " in result
    assert f"Table chunks: {expected['table_chunks']}" in result
    assert f"Image chunks: {expected['image_chunks']}" in result


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
