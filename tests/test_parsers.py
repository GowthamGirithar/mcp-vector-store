import os

import pytest

from mcp_vectordb.core.parsers import extract_text, UnsupportedFileTypeError, DocumentParseError

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


def test_extract_text_from_multi_page_pdf_returns_sequential_page_numbers():
    result = extract_text(fixture_path("sample.pdf"))

    assert len(result) == 3
    for expected_page_number, (page_number, text) in zip([1, 2, 3], result):
        assert page_number == expected_page_number
        assert isinstance(text, str)
        assert text.strip() != ""


def test_extract_text_from_txt_returns_single_tuple_with_none_page_number():
    path = fixture_path("sample.txt")
    with open(path, "r", encoding="utf-8") as f:
        expected_content = f.read()

    result = extract_text(path)

    assert result == [(None, expected_content)]


def test_extract_text_from_md_returns_single_tuple_with_none_page_number():
    path = fixture_path("sample.md")
    with open(path, "r", encoding="utf-8") as f:
        expected_content = f.read()

    result = extract_text(path)

    assert result == [(None, expected_content)]


def test_extract_text_unsupported_extension_raises_before_parsing():
    with pytest.raises(UnsupportedFileTypeError):
        extract_text(fixture_path("sample.docx"))


def test_extract_text_corrupt_pdf_raises_distinct_exception():
    with pytest.raises(DocumentParseError):
        extract_text(fixture_path("corrupt.pdf"))
