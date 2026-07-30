"""Tests for the generate_document_embedding MCP tool."""

import os

import pytest

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
async def test_generate_document_embedding_reports_chunk_counts(filename, expected):
    result = await document_embedding_tool.generate_document_embedding(
        file_path=fixture_path(filename)
    )

    assert filename in result
    assert "Text chunks: " in result
    assert f"Table chunks: {expected['table_chunks']}" in result
    assert f"Image chunks: {expected['image_chunks']}" in result
