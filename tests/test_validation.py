"""Tests for validation utilities in mcp_vectordb.utils.validation."""

import os

import pytest

from mcp_vectordb.utils.exceptions import ValidationError
from mcp_vectordb.utils.validation import (
    validate_chunk_overlap,
    validate_chunk_size,
    validate_file_path,
    validate_metadata,
    validate_text,
)


# ---------------------------------------------------------------------------
# validate_file_path
# ---------------------------------------------------------------------------


@pytest.fixture
def small_text_file(tmp_path):
    file_path = tmp_path / "sample.txt"
    file_path.write_text("hello world")
    return str(file_path)


def test_validate_file_path_happy_path(small_text_file):
    result = validate_file_path(
        small_text_file, allowed_extensions={".txt", ".md"}, max_file_size_mb=1.0
    )
    assert result == small_text_file.strip()


def test_validate_file_path_missing_file(tmp_path):
    missing = str(tmp_path / "does_not_exist.txt")
    with pytest.raises(ValidationError, match="File not found"):
        validate_file_path(missing, allowed_extensions={".txt"}, max_file_size_mb=1.0)


def test_validate_file_path_unsupported_extension(small_text_file):
    with pytest.raises(ValidationError, match="Unsupported file extension"):
        validate_file_path(
            small_text_file, allowed_extensions={".pdf", ".md"}, max_file_size_mb=1.0
        )


def test_validate_file_path_oversized_file(tmp_path):
    file_path = tmp_path / "big.txt"
    # Write ~2 MB of data.
    file_path.write_bytes(b"0" * (2 * 1024 * 1024))
    with pytest.raises(ValidationError, match="exceeds maximum"):
        validate_file_path(str(file_path), allowed_extensions={".txt"}, max_file_size_mb=1.0)


def test_validate_file_path_empty_string_raises():
    with pytest.raises(ValidationError):
        validate_file_path("", allowed_extensions={".txt"}, max_file_size_mb=1.0)


def test_validate_file_path_strips_whitespace(small_text_file):
    padded = f"  {small_text_file}  "
    result = validate_file_path(padded, allowed_extensions={".txt"}, max_file_size_mb=1.0)
    assert result == small_text_file.strip()


# ---------------------------------------------------------------------------
# validate_chunk_size
# ---------------------------------------------------------------------------


def test_validate_chunk_size_happy_path():
    assert validate_chunk_size(500) == 500


@pytest.mark.parametrize("bad_value", [0, -1, -100])
def test_validate_chunk_size_rejects_non_positive(bad_value):
    with pytest.raises(ValidationError):
        validate_chunk_size(bad_value)


@pytest.mark.parametrize("bad_value", ["500", 1.5, None])
def test_validate_chunk_size_rejects_non_int(bad_value):
    with pytest.raises(ValidationError):
        validate_chunk_size(bad_value)


# ---------------------------------------------------------------------------
# validate_chunk_overlap
# ---------------------------------------------------------------------------


def test_validate_chunk_overlap_happy_path():
    assert validate_chunk_overlap(50, chunk_size=500) == 50


def test_validate_chunk_overlap_zero_is_valid():
    assert validate_chunk_overlap(0, chunk_size=500) == 0


def test_validate_chunk_overlap_rejects_negative():
    with pytest.raises(ValidationError):
        validate_chunk_overlap(-1, chunk_size=500)


def test_validate_chunk_overlap_rejects_overlap_equal_chunk_size():
    with pytest.raises(ValidationError):
        validate_chunk_overlap(500, chunk_size=500)


def test_validate_chunk_overlap_rejects_overlap_greater_than_chunk_size():
    with pytest.raises(ValidationError):
        validate_chunk_overlap(600, chunk_size=500)


def test_validate_chunk_overlap_rejects_non_int():
    with pytest.raises(ValidationError):
        validate_chunk_overlap("50", chunk_size=500)


# ---------------------------------------------------------------------------
# validate_metadata - extended reserved keys for document upload
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "reserved_key",
    [
        "document_id",
        "source_filename",
        "file_type",
        "page_number",
        "chunk_index",
        "total_chunks",
        "uploaded_at",
    ],
)
def test_validate_metadata_rejects_new_reserved_keys(reserved_key):
    with pytest.raises(ValidationError, match="reserved"):
        validate_metadata({reserved_key: "some_value"})


def test_validate_metadata_allows_ordinary_custom_metadata():
    metadata = {"author": "jane", "category": "reports", "version": 3}
    assert validate_metadata(metadata) == metadata


# ---------------------------------------------------------------------------
# Regression: pre-existing validate_metadata / validate_text behavior
# ---------------------------------------------------------------------------


def test_validate_metadata_none_returns_none():
    assert validate_metadata(None) is None


def test_validate_metadata_rejects_non_dict():
    with pytest.raises(ValidationError, match="must be a dictionary"):
        validate_metadata("not-a-dict")


@pytest.mark.parametrize("reserved_key", ["id", "embedding", "document", "text"])
def test_validate_metadata_rejects_original_reserved_keys(reserved_key):
    with pytest.raises(ValidationError, match="reserved"):
        validate_metadata({reserved_key: "value"})


def test_validate_metadata_rejects_non_json_serializable_value():
    with pytest.raises(ValidationError, match="not JSON serializable"):
        validate_metadata({"bad": object()})


def test_validate_text_rejects_empty_string():
    with pytest.raises(ValidationError, match="empty"):
        validate_text("   ")


def test_validate_text_happy_path():
    assert validate_text("  hello  ") == "hello"
