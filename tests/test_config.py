"""Tests for DocumentConfig and its wiring into Settings."""

from mcp_vectordb.config.config import Settings, get_settings


def test_settings_has_document_config_with_defaults():
    settings = get_settings()

    assert settings.document.chunk_size == 500
    assert settings.document.chunk_overlap == 50
    assert settings.document.max_file_size_mb == 20.0


def test_from_env_document_config_defaults():
    settings = Settings.from_env()

    assert settings.document.chunk_size == 500
    assert settings.document.chunk_overlap == 50
    assert settings.document.max_file_size_mb == 20.0


def test_from_env_document_config_reads_env_overrides(monkeypatch):
    monkeypatch.setenv("DOCUMENT_CHUNK_SIZE", "1000")
    monkeypatch.setenv("DOCUMENT_CHUNK_OVERLAP", "100")
    monkeypatch.setenv("DOCUMENT_MAX_FILE_SIZE_MB", "50.5")

    settings = Settings.from_env()

    assert settings.document.chunk_size == 1000
    assert settings.document.chunk_overlap == 100
    assert settings.document.max_file_size_mb == 50.5


def test_get_settings_reflects_env_overrides(monkeypatch):
    monkeypatch.setenv("DOCUMENT_CHUNK_SIZE", "750")

    settings = get_settings()

    assert settings.document.chunk_size == 750
