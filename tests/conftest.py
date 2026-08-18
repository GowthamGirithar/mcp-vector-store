"""Shared pytest fixtures for the mcp_vectordb test suite."""

import pytest
import pytest_asyncio

import mcp_vectordb.config.config as config_module
import mcp_vectordb.services as services_module
from mcp_vectordb.adapters.factory import VectorDBFactory
from mcp_vectordb.embedding import create_embedding_service


@pytest.fixture(autouse=True)
def reset_settings_cache():
    """Reset the module-level Settings singleton before and after each test.

    ``get_settings()`` caches a single ``Settings`` instance in
    ``mcp_vectordb.config.config._settings``. Tests that mutate environment
    variables (e.g. via ``monkeypatch.setenv``) need a fresh instance to pick
    up the change, and must not leak their cached instance into later tests.
    """
    config_module._settings = None
    yield
    config_module._settings = None


@pytest_asyncio.fixture
async def real_services(tmp_path, monkeypatch):
    """Boot a real (temp-directory) Chroma adapter + local embedding service
    and install them as the ``services`` module globals that
    ``get_vector_db()``/``get_embedding_service()`` read.

    Outside of the FastMCP server lifespan (``services.setup_services``),
    those globals are never populated, so any tool that calls
    ``get_vector_db()`` gets ``None`` and blows up on first use. Tests that
    exercise a tool end-to-end need this fixture; a local Chroma path keeps
    the run hermetic (no shared state with the project's real ``chroma_db``
    directory, no network calls beyond the one-time HF model download).
    """
    settings = config_module.get_settings()
    settings.vector_db.path = str(tmp_path / "chroma_db")

    vector_db = VectorDBFactory.create_adapter(settings.vector_db)
    await vector_db.initialize()

    embedding_service = create_embedding_service(
        provider=settings.embedding.provider
        if settings.embedding.provider in ("sentence-transformers", "sentence_transformers", "local")
        else "sentence-transformers",
        model=settings.embedding.model,
        enable_cache=True,
    )

    monkeypatch.setattr(services_module, "vector_db", vector_db)
    monkeypatch.setattr(services_module, "embedding_service", embedding_service)

    yield vector_db, embedding_service

    await vector_db.close()
