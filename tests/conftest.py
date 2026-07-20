"""Shared pytest fixtures for the mcp_vectordb test suite."""

import pytest

import mcp_vectordb.config.config as config_module


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
