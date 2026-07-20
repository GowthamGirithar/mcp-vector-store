"""Tests for mcp_vectordb.config.config (Settings, DocumentConfig)."""

from mcp_vectordb.config.config import Settings


def test_document_chunk_strategy_defaults_to_recursive():
    settings = Settings.from_env()
    assert settings.document.chunk_strategy == "recursive"


def test_document_chunk_strategy_reads_env_override(monkeypatch):
    monkeypatch.setenv("DOCUMENT_CHUNK_STRATEGY", "structural")

    settings = Settings.from_env()

    assert settings.document.chunk_strategy == "structural"
