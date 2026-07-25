"""Tests for the generate_document_embedding MCP tool."""

import os

import pytest

from mcp_vectordb.tools import document_embedding as document_embedding_tool

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


@pytest.mark.asyncio
async def test_generate_document_embedding_pdf_reports_counts_per_category():
    result = await document_embedding_tool.generate_document_embedding(
        file_path=fixture_path("multimodal_sample.pdf")
    )

    assert "multimodal_sample.pdf" in result
    assert "Text chunks: " in result
    assert "Table chunks: 1" in result
    assert "Image chunks: 1" in result


@pytest.mark.asyncio
async def test_generate_document_embedding_docx_has_zero_image_chunks():
    result = await document_embedding_tool.generate_document_embedding(
        file_path=fixture_path("multimodal_sample.docx")
    )

    assert "Table chunks: 1" in result
    assert "Image chunks: 0" in result


@pytest.mark.asyncio
async def test_generate_document_embedding_pptx_has_zero_image_chunks():
    result = await document_embedding_tool.generate_document_embedding(
        file_path=fixture_path("multimodal_sample.pptx")
    )

    assert "Table chunks: 1" in result
    assert "Image chunks: 0" in result


@pytest.mark.asyncio
async def test_generate_document_embedding_md_has_zero_image_chunks():
    result = await document_embedding_tool.generate_document_embedding(
        file_path=fixture_path("multimodal_sample.md")
    )

    assert "Table chunks: 1" in result
    assert "Image chunks: 0" in result


@pytest.mark.asyncio
async def test_generate_document_embedding_unsupported_extension_raises_value_error():
    with pytest.raises(ValueError):
        await document_embedding_tool.generate_document_embedding(
            file_path=fixture_path("sample.txt")
        )


@pytest.mark.asyncio
async def test_generate_document_embedding_missing_file_raises_value_error():
    with pytest.raises(ValueError):
        await document_embedding_tool.generate_document_embedding(
            file_path=fixture_path("does_not_exist.pdf")
        )


@pytest.mark.asyncio
async def test_generate_document_embedding_corrupt_pdf_raises_runtime_error():
    with pytest.raises(RuntimeError):
        await document_embedding_tool.generate_document_embedding(
            file_path=fixture_path("corrupt.pdf")
        )


@pytest.mark.asyncio
async def test_generate_document_embedding_file_too_large_raises_value_error(
    monkeypatch, tmp_path
):
    from mcp_vectordb.config import config as config_module

    monkeypatch.setenv("DOCUMENT_MAX_FILE_SIZE_MB", "0.00001")
    config_module._settings = None

    oversized_md = tmp_path / "oversized.md"
    oversized_md.write_text("# Heading\n\nSome content that exceeds the tiny limit.\n")

    with pytest.raises(ValueError):
        await document_embedding_tool.generate_document_embedding(
            file_path=str(oversized_md)
        )
