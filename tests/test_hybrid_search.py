"""Tests for the hybrid_search tool's BM25 cache invalidation and min_score
filtering — see tools/search.py.

These mock the vector DB and BM25 layer entirely so the tests exercise only
the fusion/filtering logic in `hybrid_search`, not real embedding or BM25
scoring (covered by test_table_text.py / the BM25Index implementation
itself).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mcp_vectordb.models.document import Document, SearchResult
from mcp_vectordb.search import BM25Index
from mcp_vectordb.tools import search as search_tool


def test_invalidate_bm25_cache_removes_cached_entry():
    search_tool._bm25_cache["mycol"] = (3, BM25Index([("a", "hello world")]))
    search_tool.invalidate_bm25_cache("mycol")
    assert "mycol" not in search_tool._bm25_cache


def test_invalidate_bm25_cache_is_a_noop_for_an_uncached_collection():
    search_tool.invalidate_bm25_cache("never-cached")  # must not raise


@pytest.mark.asyncio
async def test_invalidate_bm25_cache_forces_a_rebuild_on_next_search():
    """Regression test for the BM25-cache-staleness gap: the cache used to be
    invalidated only when `document_count` changed, so an update or a
    same-size delete+insert (e.g. re-ingesting with force=True) served a
    stale index. Write-path tools now call `invalidate_bm25_cache`
    explicitly — this confirms that call actually forces `_get_bm25_index`
    to rebuild rather than serve the cached entry."""
    mock_vector_db = MagicMock()
    mock_vector_db.count_documents = AsyncMock(return_value=1)
    mock_vector_db.get_all_documents = AsyncMock(
        return_value=[Document(id="a", text="updated content", metadata={})]
    )

    first = await search_tool._get_bm25_index(mock_vector_db, "mycol")
    assert mock_vector_db.get_all_documents.await_count == 1

    # same document_count as before -> the stale count-based cache alone
    # would serve the old index without a rebuild
    second = await search_tool._get_bm25_index(mock_vector_db, "mycol")
    assert second is first
    assert mock_vector_db.get_all_documents.await_count == 1

    search_tool.invalidate_bm25_cache("mycol")
    third = await search_tool._get_bm25_index(mock_vector_db, "mycol")
    assert third is not first
    assert mock_vector_db.get_all_documents.await_count == 2


class _FakeBM25Index:
    """Stands in for a real BM25Index with fixed, known scores per doc_id."""

    def __init__(self, matches):
        self._matches = matches

    def search(self, query, top_k):
        return self._matches[:top_k]


@pytest.mark.asyncio
async def test_min_score_filters_weak_bm25_only_hits():
    """Regression test for the min_score-bypass gap: a document the vector
    leg never returned used to default to a vector_score of 1.0 and pass any
    min_score threshold unconditionally. A BM25-only hit with a weak
    normalized BM25 score must now be filtered out, while a BM25-only hit
    with a strong normalized score still passes."""
    doc_v = Document(id="doc-v", text="vector hit", metadata={})
    doc_strong = Document(id="doc-bm25-strong", text="strong bm25 hit", metadata={})
    doc_weak = Document(id="doc-bm25-weak", text="weak bm25 hit", metadata={})

    mock_vector_db = MagicMock()
    mock_vector_db.collection_exists = AsyncMock(return_value=True)
    mock_vector_db.similarity_search = AsyncMock(
        return_value=[SearchResult(document=doc_v, score=0.9)]
    )
    mock_vector_db.get_document = AsyncMock(
        side_effect=lambda doc_id, collection: {"doc-bm25-strong": doc_strong, "doc-bm25-weak": doc_weak}.get(doc_id)
    )

    fake_bm25 = _FakeBM25Index([
        ("doc-v", 5.0),
        ("doc-bm25-strong", 4.5),   # normalized 0.9 -> passes a 0.5 floor
        ("doc-bm25-weak", 0.1),     # normalized 0.02 -> fails a 0.5 floor
    ])

    mock_embedding_service = MagicMock()
    mock_embedding_service.generate_embedding = AsyncMock(return_value=[0.0] * 384)

    mock_settings = MagicMock()
    mock_settings.search.default_top_k = 10
    mock_settings.search.default_min_score = 0.5
    mock_settings.search.use_reranker = False
    mock_settings.search.rrf_k = 60
    mock_settings.search.vector_weight = 1.0
    mock_settings.search.bm25_weight = 1.0

    with patch.object(search_tool, "get_vector_db", return_value=mock_vector_db), \
         patch.object(search_tool, "get_embedding_service", return_value=mock_embedding_service), \
         patch.object(search_tool, "get_settings", return_value=mock_settings), \
         patch.object(search_tool, "_get_bm25_index", new=AsyncMock(return_value=fake_bm25)):
        results = await search_tool.hybrid_search(query="test query", collection="documents")

    assert doc_v.text in results
    assert doc_strong.text in results
    assert doc_weak.text not in results
