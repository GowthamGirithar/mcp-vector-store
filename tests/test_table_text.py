"""Unit tests for `chunking.table_text.linearize_table_html` (gap: table text
dropped — see `chunking/process_document.py`'s Table branch, which used to
route table HTML into `tableHTML` only, never into the embedded/searched
text)."""

from mcp_vectordb.chunking.table_text import linearize_table_html


def test_linearizes_thead_tbody_table_with_header_value_pairs():
    html = (
        "<table><thead><tr><th>Name</th><th>Score</th></tr></thead>"
        "<tbody><tr><td>Alice</td><td>92</td></tr>"
        "<tr><td>Bob</td><td>85</td></tr></tbody></table>"
    )
    text = linearize_table_html(html)
    assert "Name: Alice" in text
    assert "Score: 92" in text
    assert "Name: Bob" in text
    assert "Score: 85" in text


def test_detects_header_row_without_thead():
    html = "<table><tr><td>Name</td><td>Score</td></tr><tr><td>Alice</td><td>92</td></tr></table>"
    text = linearize_table_html(html)
    assert "Name: Alice" in text
    assert "Score: 92" in text


def test_all_numeric_first_row_is_not_treated_as_header():
    html = "<table><tr><td>1</td><td>2</td></tr><tr><td>3</td><td>4</td></tr></table>"
    text = linearize_table_html(html)
    # no header detected -> plain pipe-joined rows, no "Header: value" pairing
    assert text == "1 | 2\n3 | 4"


def test_empty_and_malformed_input_returns_empty_string():
    assert linearize_table_html("") == ""
    assert linearize_table_html("<table></table>") == ""
    # unclosed tags must not raise
    assert linearize_table_html("<table><tr><td>Name<td>Score") != None
