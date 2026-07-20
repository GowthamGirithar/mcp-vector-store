import pytest

from mcp_vectordb.core.chunking import recursive_chunk


def test_short_text_returns_single_chunk():
    text = "This is a short piece of text."
    chunks = recursive_chunk(text, chunk_size=1000, chunk_overlap=100)
    assert chunks == [text]


def test_paragraph_split_respects_boundaries():
    para1 = "A" * 60
    para2 = "B" * 60
    para3 = "C" * 60
    text = "\n\n".join([para1, para2, para3])
    chunks = recursive_chunk(text, chunk_size=70, chunk_overlap=10)
    assert len(chunks) > 1
    # Each paragraph should be fully contained in some chunk since each
    # paragraph (60 chars) fits within chunk_size (70) on its own.
    for para in (para1, para2, para3):
        assert any(para in chunk for chunk in chunks)


def test_line_split_when_paragraph_still_too_long():
    line1 = "x" * 30
    line2 = "y" * 30
    line3 = "z" * 30
    # Single paragraph (no blank-line breaks) but multiple lines.
    text = "\n".join([line1, line2, line3])
    chunks = recursive_chunk(text, chunk_size=40, chunk_overlap=5)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 40 or "\n" not in chunk


def test_sentence_split_when_line_still_too_long():
    sentence1 = "This is sentence one and it is fairly long indeed"
    sentence2 = "This is sentence two and it is also fairly long"
    text = sentence1 + ". " + sentence2 + "."
    chunks = recursive_chunk(text, chunk_size=55, chunk_overlap=5)
    assert len(chunks) > 1


def test_hard_window_fallback_no_separators():
    text = "a" * 500
    chunk_size = 50
    chunk_overlap = 10
    chunks = recursive_chunk(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= chunk_size
        assert chunk != ""


def test_hard_window_fallback_overlap_is_reproduced():
    text = "0123456789" * 60  # 600 unique-ish repeating chars, no separators
    chunk_size = 50
    chunk_overlap = 10
    chunks = recursive_chunk(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    assert len(chunks) > 1
    for i in range(len(chunks) - 1):
        tail = chunks[i][-chunk_overlap:]
        assert tail in chunks[i + 1]


def test_hard_window_terminates_on_pathological_input():
    text = "q" * 100_000
    chunks = recursive_chunk(text, chunk_size=1000, chunk_overlap=200)
    assert len(chunks) > 1
    assert all(chunk != "" for chunk in chunks)


def test_chunk_overlap_greater_equal_chunk_size_raises():
    with pytest.raises(ValueError):
        recursive_chunk("some text", chunk_size=100, chunk_overlap=100)
    with pytest.raises(ValueError):
        recursive_chunk("some text", chunk_size=100, chunk_overlap=150)


def test_chunk_size_non_positive_raises():
    with pytest.raises(ValueError):
        recursive_chunk("some text", chunk_size=0, chunk_overlap=0)
    with pytest.raises(ValueError):
        recursive_chunk("some text", chunk_size=-10, chunk_overlap=0)


def test_chunk_overlap_negative_raises():
    with pytest.raises(ValueError):
        recursive_chunk("some text", chunk_size=100, chunk_overlap=-1)


def test_empty_string_returns_empty_list():
    assert recursive_chunk("", chunk_size=100, chunk_overlap=10) == []


def test_no_chunk_is_ever_empty_string():
    text = "\n\n".join(["", "A" * 10, "", "B" * 10, ""])
    chunks = recursive_chunk(text, chunk_size=20, chunk_overlap=2)
    assert all(chunk != "" for chunk in chunks)
