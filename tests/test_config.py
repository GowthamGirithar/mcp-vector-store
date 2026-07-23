"""Tests for mcp_vectordb.config.config (Settings, DocumentConfig)."""

from mcp_vectordb.config.config import Settings


def test_document_config_defaults_chunk_size():
    settings = Settings.from_env()
    assert settings.document.chunk_size == 500
