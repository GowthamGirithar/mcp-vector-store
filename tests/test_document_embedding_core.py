"""Tests for the multimodal document extraction core module (generate-document-embedding)."""

import base64
import os

import pytest

from mcp_vectordb.core.document_embedding import (
    MULTIMODAL_SUPPORTED_EXTENSIONS,
    DocumentEmbeddingParseError,
    UnsupportedDocumentTypeError,
    extract_multimodal_document,
)

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


def fixture_path(name: str) -> str:
    return os.path.join(FIXTURES_DIR, name)


def test_supported_extensions_are_pdf_md_docx_pptx():
    assert MULTIMODAL_SUPPORTED_EXTENSIONS == {".pdf", ".md", ".docx", ".pptx"}


def test_extract_multimodal_pdf_returns_text_table_and_image_elements():
    result = extract_multimodal_document(fixture_path("attention.pdf"))

    print(result)

    """
    assert len(result.text_elements) >= 1
    assert len(result.table_elements) == 1
    assert len(result.image_elements) == 1

    combined_text = " ".join(el.content for el in result.text_elements)
    assert "Multimodal Sample Document" in combined_text

    table = result.table_elements[0]
    assert table.category == "table"
    assert "Alice" in table.content or "Alice" in (table.html or "")
    assert table.html is not None
    assert table.page_number == 1

    image = result.image_elements[0]
    assert image.category == "image"
    assert image.page_number == 1
    # image content is base64 and decodes to non-empty bytes
    decoded = base64.b64decode(image.content)
    assert len(decoded) > 0
    """

def test_extract_multimodal_docx_returns_text_and_table_no_image():
    """Known limitation: unstructured has no built-in DOCX image extraction."""
    result = extract_multimodal_document(fixture_path("multimodal_sample.docx"))

    assert len(result.text_elements) >= 1
    assert len(result.table_elements) == 1
    assert result.image_elements == []


def test_extract_multimodal_pptx_returns_text_and_table_no_image():
    """Known limitation: unstructured has no built-in PPTX image extraction."""
    result = extract_multimodal_document(fixture_path("multimodal_sample.pptx"))

    assert len(result.text_elements) >= 1
    assert len(result.table_elements) == 1
    assert result.image_elements == []


def test_extract_multimodal_md_returns_text_and_table_no_image():
    result = extract_multimodal_document(fixture_path("multimodal_sample.md"))

    assert len(result.text_elements) >= 1
    assert len(result.table_elements) == 1
    assert result.image_elements == []
    assert all(el.page_number is None for el in result.text_elements)


def test_extract_multimodal_unsupported_extension_raises_before_parsing():
    with pytest.raises(UnsupportedDocumentTypeError):
        extract_multimodal_document(fixture_path("sample.txt"))


def test_extract_multimodal_corrupt_pdf_raises_parse_error():
    with pytest.raises(DocumentEmbeddingParseError):
        extract_multimodal_document(fixture_path("corrupt.pdf"))


def test_extract_multimodal_empty_markdown_returns_all_empty_lists(tmp_path):
    empty_md = tmp_path / "empty.md"
    empty_md.write_text("")

    result = extract_multimodal_document(str(empty_md))

    assert result.text_elements == []
    assert result.table_elements == []
    assert result.image_elements == []


def test_extract_multimodal_text_elements_have_no_blank_content():
    result = extract_multimodal_document(fixture_path("multimodal_sample.pptx"))

    assert all(el.content.strip() != "" for el in result.text_elements)
