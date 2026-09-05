"""Tests for the silent-partial-ingestion gap: `_process_pdf_range` used to
swallow a per-range exception and return an empty list with no signal to the
caller, so `generate_document_embedding` could report "Successfully
processed" on a document that was actually half-indexed. See
`chunking/process_document.py`'s `pages_failed` parameter and
`tools/document_embedding.py`'s use of it."""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_vectordb.chunking.process_document import _process_pdf_range, process_document
from mcp_vectordb.tools import document_embedding as document_embedding_tool

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


def test_process_pdf_range_reports_failure_instead_of_hiding_it():
    with patch(
        "mcp_vectordb.chunking.process_document.partition_pdf",
        side_effect=RuntimeError("layout model crashed"),
    ):
        elements, failed = _process_pdf_range(fixture_path("multimodal_sample.pdf"), 1, 1)

    assert elements == []
    assert failed is True


def test_process_pdf_range_reports_success_normally():
    real_elements = ["element-a", "element-b"]
    with patch(
        "mcp_vectordb.chunking.process_document.partition_pdf",
        return_value=real_elements,
    ):
        elements, failed = _process_pdf_range(fixture_path("multimodal_sample.pdf"), 1, 1)

    assert elements == real_elements
    assert failed is False


def test_process_document_populates_pages_failed_on_small_pdf():
    """The `multimodal_sample.pdf` fixture is under the 30-page threshold, so
    it takes the sequential (non-multiprocessing) path — the simplest case
    to force a failure through."""
    with patch(
        "mcp_vectordb.chunking.process_document.partition_pdf",
        side_effect=RuntimeError("layout model crashed"),
    ):
        pages_failed = []
        results = process_document(
            fixture_path("multimodal_sample.pdf"), parser="unstructured", pages_failed=pages_failed
        )

    assert results == []
    assert pages_failed == [(1, 1)]


def test_process_document_pages_failed_defaults_to_silent_when_not_passed():
    """Backward compatibility: a caller that doesn't care about partial
    failures (the default) sees the same behavior as before — no exception,
    just fewer elements."""
    with patch(
        "mcp_vectordb.chunking.process_document.partition_pdf",
        side_effect=RuntimeError("layout model crashed"),
    ):
        results = process_document(fixture_path("multimodal_sample.pdf"), parser="unstructured")

    assert results == []


@pytest.mark.asyncio
async def test_generate_document_embedding_reports_partial_failure_instead_of_success():
    """End-to-end: when process_document reports a failed page range, the
    tool's returned message must say so and must not claim success."""
    mock_vector_db = MagicMock()
    mock_vector_db.collection_exists = AsyncMock(return_value=True)
    mock_vector_db.get_all_documents = AsyncMock(return_value=[])
    mock_vector_db.store_documents = AsyncMock(return_value=["chunk-id"])

    mock_embedding_service = MagicMock()
    mock_embedding_service.dimension = 384
    mock_embedding_service.count_tokens = lambda text: len(text.split())
    mock_embedding_service.max_input_tokens = 256
    mock_embedding_service.generate_embeddings = AsyncMock(return_value=[[0.0] * 384])

    def fake_process_document(*args, **kwargs):
        pages_failed = kwargs.get("pages_failed")
        if pages_failed is not None:
            pages_failed.append((5, 8))
        return [MagicMock(text="surviving chunk text", tableHTML=[], imageBase64=[])]

    with patch.object(document_embedding_tool, "process_document", side_effect=fake_process_document), \
         patch.object(document_embedding_tool, "get_vector_db", return_value=mock_vector_db), \
         patch.object(document_embedding_tool, "get_embedding_service", return_value=mock_embedding_service):
        result = await document_embedding_tool.generate_document_embedding(
            file_path=fixture_path("multimodal_sample.md")
        )

    assert "Successfully processed" not in result
    assert "Partially processed" in result
    assert "5-8" in result
    assert "PARTIALLY indexed" in result
    mock_vector_db.store_documents.assert_awaited_once()


@pytest.mark.asyncio
async def test_generate_document_embedding_reports_total_failure_when_no_chunks_survive():
    mock_vector_db = MagicMock()
    mock_vector_db.collection_exists = AsyncMock(return_value=True)
    mock_vector_db.get_all_documents = AsyncMock(return_value=[])

    mock_embedding_service = MagicMock()
    mock_embedding_service.count_tokens = lambda text: len(text.split())
    mock_embedding_service.max_input_tokens = 256

    def fake_process_document(*args, **kwargs):
        pages_failed = kwargs.get("pages_failed")
        if pages_failed is not None:
            pages_failed.append((1, 10))
        return []

    with patch.object(document_embedding_tool, "process_document", side_effect=fake_process_document), \
         patch.object(document_embedding_tool, "get_vector_db", return_value=mock_vector_db), \
         patch.object(document_embedding_tool, "get_embedding_service", return_value=mock_embedding_service):
        result = await document_embedding_tool.generate_document_embedding(
            file_path=fixture_path("multimodal_sample.md")
        )

    assert "FAILED to process" in result
    assert "1-10" in result
