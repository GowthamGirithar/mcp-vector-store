import pytest

from mcp_vectordb.core.chunking import recursive_chunk, chunk_data


SHORT_TEXT = "This is a short piece of text."

PARAGRAPH_A = "Alpha team finished the quarterly report ahead of schedule this week."
PARAGRAPH_B = "Bravo squad is still validating migration scripts before the rollout."
PARAGRAPH_C = "Charlie group will present the new pricing model on Friday morning."
PARAGRAPH_CHUNK_SIZE = max(len(PARAGRAPH_A), len(PARAGRAPH_B), len(PARAGRAPH_C)) + 5

LINE_X = "Deploy the staging environment and run the smoke test suite first."
LINE_Y = "Confirm the database migration rollback plan with the platform team."
LINE_Z = "Notify the on-call engineer once the canary release finishes."
LINE_CHUNK_SIZE = max(len(LINE_X), len(LINE_Y), len(LINE_Z)) + 5

WORD_TEXT = "The mission control team monitored telemetry throughout the entire launch window closely"

HARD_WINDOW_WORD = "pneumonoultramicroscopicsilicovolcanoconiosis"
HARD_WINDOW_TEXT = HARD_WINDOW_WORD * 2

OVERLAP_WORD = "supercalifragilisticexpialidocious"
OVERLAP_TEXT = OVERLAP_WORD * 3

PATHOLOGICAL_WORD = "antidisestablishmentarianism"
PATHOLOGICAL_TEXT = PATHOLOGICAL_WORD * 10

WORD_A = "alphabetic"
WORD_B = "brontosaur"


CASES = {
    "short_text_single_chunk": {
        "description": "Text shorter than chunk_size returns as a single chunk",
        "text": SHORT_TEXT,
        "chunk_size": 1000,
        "chunk_overlap": 100,
        "expected_chunks": [SHORT_TEXT],
    },
    "paragraph_split_respects_boundaries": {
        "description": "Paragraphs that individually fit chunk_size stay intact in some chunk",
        "text": "\n\n".join([PARAGRAPH_A, PARAGRAPH_B, PARAGRAPH_C]),
        "chunk_size": PARAGRAPH_CHUNK_SIZE,
        "chunk_overlap": 10,
        "expected_chunks": [PARAGRAPH_A, PARAGRAPH_B, PARAGRAPH_C],
    },
    "line_split_when_paragraph_too_long": {
        "description": "Falls back to line splitting when a paragraph exceeds chunk_size",
        "text": "\n".join([LINE_X, LINE_Y, LINE_Z]),
        "chunk_size": LINE_CHUNK_SIZE,
        "chunk_overlap": 5,
        "expected_chunks": [LINE_X, LINE_Y, LINE_Z],
    },
    "word_split_when_line_too_long": {
        "description": "Falls back to word (space) splitting when a line exceeds chunk_size",
        "text": WORD_TEXT,
        "chunk_size": 30,
        "chunk_overlap": 5,
        "expected_chunks": [
            "The mission control team",
            "monitored telemetry throughout",
            "the entire launch window",
            "closely",
        ],
    },
    "hard_window_fallback_no_separators": {
        "description": "A single word longer than chunk_size forces hard-window slicing",
        "text": HARD_WINDOW_TEXT,
        "chunk_size": 50,
        "chunk_overlap": 10,
        "expected_chunks": [
            "pneumonoultramicroscopicsilicovolcanoconiosispneum",
            "iosispneumonoultramicroscopicsilicovolcanoconiosis",
        ],
    },
    "hard_window_overlap_reproduced": {
        "description": "Hard-window fallback reproduces chunk_overlap tail in the next chunk",
        "text": OVERLAP_TEXT,
        "chunk_size": 50,
        "chunk_overlap": 10,
        "expected_chunks": [
            "supercalifragilisticexpialidocioussupercalifragili",
            "alifragilisticexpialidocioussupercalifragilisticex",
            "gilisticexpialidocious",
        ],
    },
    "hard_window_terminates_on_pathological_input": {
        "description": "Large single-word input still terminates and yields non-empty chunks",
        "text": PATHOLOGICAL_TEXT,
        "chunk_size": 100,
        "chunk_overlap": 20,
        "expected_chunks": [
            "antidisestablishmentarianismantidisestablishmentarianismantidisestablishmentarianismantidisestablish",
            "nismantidisestablishmentarianismantidisestablishmentarianismantidisestablishmentarianismantidisestab",
            "arianismantidisestablishmentarianismantidisestablishmentarianismantidisestablishmentarianismantidise",
            "mentarianismantidisestablishmentarianism",
        ],
    },
    "empty_string_returns_empty_list": {
        "description": "Empty input returns an empty chunk list",
        "text": "",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "expected_chunks": [],
    },
    "no_chunk_is_ever_empty": {
        "description": "Blank paragraphs in the input never produce empty-string chunks",
        "text": "\n\n".join(["", WORD_A, "", WORD_B, ""]),
        "chunk_size": 20,
        "chunk_overlap": 2,
        "expected_chunks": [WORD_A, WORD_B],
    },
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_recursive_chunk(case):
    chunks = recursive_chunk(
        text=case["text"],
        chunk_size=case["chunk_size"],
        chunk_overlap=case["chunk_overlap"],
    )
    assert chunks == case["expected_chunks"], case["description"]


# ---------------------------------------------------------------------------
# chunk_data
# ---------------------------------------------------------------------------
#
# chunk_data implements the unified auto-chunking pipeline, mirroring
# five explicit stages:
#   1/2. Decide header-based splitting (Markdown with headings) vs
#        layout-aware extraction (per-page: PDFs, headingless text), and
#        split into sections accordingly.
#   3/4. Any section whose approximate size exceeds chunk_size is
#        recursively split with 10% overlap; smaller sections are kept
#        whole.
#   5. Every returned chunk carries its own breadcrumb metadata
#      (source_filename > heading_path > page N).
#
# Each result is a Chunk(page_number, text, breadcrumb) namedtuple.

FILENAME = "sample.md"


def test_chunk_document_markdown_with_headings_produces_one_chunk_per_section():
    text = (
        "## Section One\n"
        "First section body.\n"
        "## Section Two\n"
        "Second section body.\n"
        "## Section Three\n"
        "Third section body.\n"
    )

    chunks = chunk_data([(None, text)], chunk_size=1000, source_filename=FILENAME)

    assert len(chunks) == 3
    assert [c.page_number for c in chunks] == [None, None, None]
    assert chunks[0].text.startswith("## Section One")
    assert chunks[1].text.startswith("## Section Two")
    assert chunks[2].text.startswith("## Section Three")


def test_chunk_document_nested_headings_produce_hierarchical_breadcrumb():
    text = (
        "# Intro\n"
        "Intro body.\n"
        "## Background\n"
        "Background body.\n"
        "# Conclusion\n"
        "Conclusion body.\n"
    )

    chunks = chunk_data([(None, text)], chunk_size=1000, source_filename=FILENAME)

    breadcrumbs = [c.breadcrumb for c in chunks]
    assert breadcrumbs == [
        f"{FILENAME} > Intro",
        f"{FILENAME} > Intro > Background",
        f"{FILENAME} > Conclusion",
    ]


def test_chunk_document_markdown_with_no_headings_is_kept_as_single_section():
    text = "\n\n".join([PARAGRAPH_A, PARAGRAPH_B, PARAGRAPH_C])

    chunks = chunk_data([(None, text)], chunk_size=1000, source_filename=FILENAME)

    assert chunks == [(None, text, FILENAME)]


def test_chunk_document_oversized_section_is_recursively_split_with_ten_percent_overlap():
    heading = "## Only Section\n"
    body = " ".join(["word"] * 400)
    text = heading + body

    chunks = chunk_data([(None, text)], chunk_size=50, source_filename=FILENAME)

    assert len(chunks) > 1
    assert all(c.page_number is None for c in chunks)
    assert all(c.breadcrumb == f"{FILENAME} > Only Section" for c in chunks)
    expected_pieces = recursive_chunk(text, chunk_size=50, chunk_overlap=5)
    assert [c.text for c in chunks] == expected_pieces


def test_chunk_document_multi_page_input_treats_each_page_as_a_section():
    pages = [
        (1, "Page one content, short."),
        (2, "Page two content, short."),
        (3, "Page three content, short."),
    ]

    chunks = chunk_data(pages, chunk_size=1000, source_filename="sample.pdf")

    assert chunks == [
        (1, "Page one content, short.", "sample.pdf > page 1"),
        (2, "Page two content, short.", "sample.pdf > page 2"),
        (3, "Page three content, short.", "sample.pdf > page 3"),
    ]


def test_chunk_document_oversized_page_is_recursively_split():
    pages = [(1, " ".join(["word"] * 400))]

    chunks = chunk_data(pages, chunk_size=50, source_filename="sample.pdf")

    assert len(chunks) > 1
    assert all(c.page_number == 1 for c in chunks)
    assert all(c.breadcrumb == "sample.pdf > page 1" for c in chunks)


def test_chunk_document_empty_input_returns_empty_list():
    assert chunk_data([], chunk_size=100, source_filename=FILENAME) == []


def test_chunk_document_single_page_empty_text_returns_empty_list():
    assert chunk_data([(None, "")], chunk_size=100, source_filename=FILENAME) == []
