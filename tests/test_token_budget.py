"""Unit tests for `chunking.token_budget` (gap: embedding truncation — chunks
produced by `chunk_by_title` were sized in characters with no knowledge of
the embedding model's token limit, so long chunks were silently truncated at
embedding time; see `chunking/process_document.py`'s `count_tokens`/
`max_tokens` parameters)."""

from mcp_vectordb.chunking.token_budget import fit_chunks_to_budget, split_text_to_budget


def _word_count(text: str) -> int:
    return len(text.split())


def test_text_within_budget_is_returned_unsplit():
    assert split_text_to_budget("one two three", _word_count, budget=10) == ["one two three"]


def test_oversize_text_is_split_on_sentence_boundaries_within_budget():
    text = "One two three. Four five six. Seven eight nine. Ten eleven twelve."
    pieces = split_text_to_budget(text, _word_count, budget=6, overlap=2)
    assert len(pieces) > 1
    for piece in pieces:
        assert _word_count(piece) <= 6
    # every sentence's content is preserved somewhere in the output
    for word in ("One", "six", "nine", "twelve"):
        assert any(word in piece for piece in pieces)


def test_single_run_on_sentence_over_budget_is_hard_split_on_words():
    long_sentence = " ".join(f"w{i}" for i in range(20)) + "."
    pieces = split_text_to_budget(long_sentence, _word_count, budget=5)
    assert len(pieces) > 1
    for piece in pieces:
        assert _word_count(piece) <= 5
    assert "w0" in pieces[0]
    assert "w19" in pieces[-1]


def test_empty_and_blank_text_returns_no_pieces():
    assert split_text_to_budget("", _word_count, budget=5) == []
    assert split_text_to_budget("   ", _word_count, budget=5) == []


def test_fit_chunks_to_budget_splits_oversize_and_carries_table_only_on_first_piece():
    long_text = " ".join(f"w{i}" for i in range(20)) + "."
    chunks = [(long_text, ["<table>t</table>"], ["base64img"])]
    fitted = fit_chunks_to_budget(chunks, _word_count, budget=5, min_tokens=1)
    assert len(fitted) > 1
    assert fitted[0][1] == ["<table>t</table>"]
    assert fitted[0][2] == ["base64img"]
    for text, tables, images in fitted[1:]:
        assert tables == []
        assert images == []


def test_fit_chunks_to_budget_merges_undersize_chunks_within_budget():
    chunks = [("a", [], []), ("b", [], []), ("this one is long enough to not merge further", [], [])]
    fitted = fit_chunks_to_budget(chunks, _word_count, budget=50, min_tokens=3)
    # "a" and "b" (1 token each) should merge into one piece under min_tokens=3
    assert len(fitted) < len(chunks)
