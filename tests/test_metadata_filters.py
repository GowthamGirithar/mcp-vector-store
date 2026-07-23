"""Integration tests for metadata filtering in similarity_search."""

import pytest
import pytest_asyncio

from mcp_vectordb.adapters.chroma import ChromaAdapter
from mcp_vectordb.config.config import VectorDBConfig
from mcp_vectordb.core.document import Document


@pytest_asyncio.fixture
async def adapter(tmp_path):
    config = VectorDBConfig(path=str(tmp_path / "chroma_db"))
    db = ChromaAdapter(config)
    await db.initialize()
    yield db
    await db.close()


async def _seed(db, collection="filters_test"):
    await db.create_collection(collection, dimension=2)
    documents = [
        Document(id="1", text="doc1", embedding=[1.0, 0.0], metadata={"tenant": "a", "type": "faq"}),
        Document(id="2", text="doc2", embedding=[1.0, 0.0], metadata={"tenant": "a", "type": "policy"}),
        Document(id="3", text="doc3", embedding=[1.0, 0.0], metadata={"tenant": "b", "type": "faq"}),
    ]
    await db.store_documents(documents, collection)


@pytest.mark.asyncio
async def test_single_key_filter(adapter):
    await _seed(adapter)
    results = await adapter.similarity_search(
        query_embedding=[1.0, 0.0],
        collection="filters_test",
        top_k=10,
        filters={"tenant": "a"},
    )
    assert {r.document.id for r in results} == {"1", "2"}


@pytest.mark.asyncio
async def test_multi_key_filter_is_anded(adapter):
    await _seed(adapter)
    results = await adapter.similarity_search(
        query_embedding=[1.0, 0.0],
        collection="filters_test",
        top_k=10,
        filters={"tenant": "a", "type": "faq"},
    )
    assert {r.document.id for r in results} == {"1"}


@pytest.mark.asyncio
async def test_no_filter_returns_all(adapter):
    await _seed(adapter)
    results = await adapter.similarity_search(
        query_embedding=[1.0, 0.0],
        collection="filters_test",
        top_k=10,
        filters=None,
    )
    assert {r.document.id for r in results} == {"1", "2", "3"}
