"""Tests for the store_document MCP tool (document-upload-tool T5)."""

import os

import pytest

from mcp_vectordb.core.document import Document
from mcp_vectordb.tools import storage as storage_tool

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


class FakeEmbeddingService:
    """Fake embedding service that returns deterministic fixed-dimension vectors."""

    def __init__(self, dimension: int = 4, fail: bool = False):
        self._dimension = dimension
        self.fail = fail

    async def generate_embedding(self, text):
        return [0.1] * self._dimension

    async def generate_embeddings(self, texts):
        if self.fail:
            raise RuntimeError("embedding backend unavailable")
        return [[0.1] * self._dimension for _ in texts]

    @property
    def dimension(self):
        return self._dimension

    @property
    def model_name(self):
        return "fake-model"


class FakeVectorDB:
    """Fake vector DB adapter that stores documents in memory."""

    def __init__(self, fail_store: bool = False):
        self.collections = set()
        self.stored = {}
        self.fail_store = fail_store

    async def collection_exists(self, name):
        return name in self.collections

    async def create_collection(self, name, dimension, metadata=None):
        self.collections.add(name)
        return True

    async def store_documents(self, documents, collection):
        if self.fail_store:
            raise RuntimeError("storage backend unavailable")
        self.stored.setdefault(collection, []).extend(documents)
        return [d.id for d in documents]


@pytest.fixture
def fake_vector_db():
    return FakeVectorDB()


@pytest.fixture
def fake_embedding_service():
    return FakeEmbeddingService()


@pytest.fixture(autouse=True)
def patch_services(monkeypatch, fake_vector_db, fake_embedding_service):
    monkeypatch.setattr(storage_tool, "get_vector_db", lambda: fake_vector_db)
    monkeypatch.setattr(
        storage_tool, "get_embedding_service", lambda: fake_embedding_service
    )


@pytest.mark.asyncio
async def test_upload_multi_page_pdf_produces_correct_chunk_metadata(fake_vector_db):
    result = await storage_tool.store_document(
        file_path=fixture_path("sample.pdf"),
        collection="documents",
    )

    stored = fake_vector_db.stored["documents"]
    assert len(stored) > 0

    document_ids = {doc.metadata["document_id"] for doc in stored}
    assert len(document_ids) == 1
    document_id = next(iter(document_ids))

    total_chunks = len(stored)
    for expected_index, doc in enumerate(stored):
        assert doc.metadata["chunk_index"] == expected_index
        assert doc.metadata["total_chunks"] == total_chunks
        assert doc.metadata["document_id"] == document_id
        assert doc.metadata["source_filename"] == "sample.pdf"
        assert doc.metadata["file_type"] == ".pdf"
        assert doc.metadata["page_number"] in (1, 2, 3)
        assert doc.embedding == [0.1] * 4

    assert document_id in result
    assert str(total_chunks) in result


@pytest.mark.asyncio
async def test_upload_txt_file_has_none_page_number(fake_vector_db):
    await storage_tool.store_document(
        file_path=fixture_path("sample.txt"),
        collection="documents",
    )

    stored = fake_vector_db.stored["documents"]
    assert len(stored) > 0
    for doc in stored:
        assert doc.metadata["page_number"] is None
        assert doc.metadata["file_type"] == ".txt"


@pytest.mark.asyncio
async def test_upload_md_file_has_none_page_number(fake_vector_db):
    await storage_tool.store_document(
        file_path=fixture_path("sample.md"),
        collection="documents",
    )

    stored = fake_vector_db.stored["documents"]
    assert len(stored) > 0
    for doc in stored:
        assert doc.metadata["page_number"] is None
        assert doc.metadata["file_type"] == ".md"


@pytest.mark.asyncio
async def test_upload_unsupported_extension_raises_value_error():
    with pytest.raises(ValueError):
        await storage_tool.store_document(file_path=fixture_path("sample.docx"))


@pytest.mark.asyncio
async def test_upload_file_exceeding_max_size_raises_value_error(monkeypatch):
    # Force a tiny max_file_size_mb so the real fixture file exceeds it.
    from mcp_vectordb.config.config import get_settings

    settings = get_settings()
    monkeypatch.setattr(settings.document, "max_file_size_mb", 0.0000001)

    with pytest.raises(ValueError):
        await storage_tool.store_document(file_path=fixture_path("sample.txt"))


@pytest.mark.asyncio
async def test_upload_reserved_metadata_key_raises_value_error():
    with pytest.raises(ValueError):
        await storage_tool.store_document(
            file_path=fixture_path("sample.txt"),
            metadata={"document_id": "caller-supplied"},
        )


@pytest.mark.asyncio
async def test_upload_corrupt_pdf_raises_runtime_error():
    with pytest.raises(RuntimeError):
        await storage_tool.store_document(file_path=fixture_path("corrupt.pdf"))


@pytest.mark.asyncio
async def test_upload_embedding_failure_surfaces_as_runtime_error(monkeypatch):
    failing_embedding_service = FakeEmbeddingService(fail=True)
    monkeypatch.setattr(
        storage_tool, "get_embedding_service", lambda: failing_embedding_service
    )

    with pytest.raises(RuntimeError):
        await storage_tool.store_document(file_path=fixture_path("sample.txt"))


@pytest.mark.asyncio
async def test_upload_vector_db_store_failure_surfaces_as_runtime_error(monkeypatch):
    failing_vector_db = FakeVectorDB(fail_store=True)
    monkeypatch.setattr(storage_tool, "get_vector_db", lambda: failing_vector_db)

    with pytest.raises(RuntimeError):
        await storage_tool.store_document(file_path=fixture_path("sample.txt"))


@pytest.mark.asyncio
async def test_upload_document_id_is_auto_generated_and_shared_across_chunks(
    fake_vector_db,
):
    await storage_tool.store_document(file_path=fixture_path("sample.txt"))

    stored = fake_vector_db.stored["documents"]
    document_ids = {doc.metadata["document_id"] for doc in stored}
    assert len(document_ids) == 1


# ---------------------------------------------------------------------------
# chunk_strategy (T6)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upload_structural_strategy_on_markdown_produces_section_chunks(
    fake_vector_db,
):
    await storage_tool.store_document(
        file_path=fixture_path("sample_sections.md"),
        chunk_strategy="structural",
    )

    stored = fake_vector_db.stored["documents"]
    assert len(stored) == 3
    assert stored[0].text.startswith("# Introduction")
    assert stored[1].text.startswith("## Background")
    assert stored[2].text.startswith("## Details")
    for doc in stored:
        assert doc.metadata["page_number"] is None


@pytest.mark.asyncio
async def test_upload_structural_strategy_on_multi_page_pdf_produces_one_chunk_per_page(
    fake_vector_db,
):
    await storage_tool.store_document(
        file_path=fixture_path("sample.pdf"),
        chunk_strategy="structural",
    )

    stored = fake_vector_db.stored["documents"]
    assert len(stored) == 3
    page_numbers = [doc.metadata["page_number"] for doc in stored]
    assert page_numbers == [1, 2, 3]


@pytest.mark.asyncio
async def test_upload_omitted_chunk_strategy_defaults_to_recursive(fake_vector_db):
    await storage_tool.store_document(file_path=fixture_path("sample.txt"))

    stored = fake_vector_db.stored["documents"]
    assert len(stored) > 0


@pytest.mark.asyncio
async def test_upload_invalid_chunk_strategy_raises_value_error_before_parsing(
    monkeypatch,
):
    called = {"extract_text": False}

    def fake_extract_text(*args, **kwargs):
        called["extract_text"] = True
        raise AssertionError("extract_text should not be called")

    monkeypatch.setattr(storage_tool, "extract_text", fake_extract_text)

    with pytest.raises(ValueError):
        await storage_tool.store_document(
            file_path=fixture_path("sample.txt"),
            chunk_strategy="bogus",
        )

    assert called["extract_text"] is False


@pytest.mark.asyncio
async def test_upload_structural_strategy_on_txt_matches_recursive_output(
    fake_vector_db,
):
    await storage_tool.store_document(
        file_path=fixture_path("sample.txt"),
        collection="structural_collection",
        chunk_strategy="structural",
    )
    structural_chunks = [
        doc.text for doc in fake_vector_db.stored["structural_collection"]
    ]

    await storage_tool.store_document(
        file_path=fixture_path("sample.txt"),
        collection="recursive_collection",
        chunk_strategy="recursive",
    )
    recursive_chunks = [
        doc.text for doc in fake_vector_db.stored["recursive_collection"]
    ]

    assert structural_chunks == recursive_chunks
