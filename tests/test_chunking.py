import pytest

from mcp_vectordb.core.chunking import recursive_chunk


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
        "expected_error": None,
    },
    "paragraph_split_respects_boundaries": {
        "description": "Paragraphs that individually fit chunk_size stay intact in some chunk",
        "text": "\n\n".join([PARAGRAPH_A, PARAGRAPH_B, PARAGRAPH_C]),
        "chunk_size": PARAGRAPH_CHUNK_SIZE,
        "chunk_overlap": 10,
        "expected_chunks": [PARAGRAPH_A, PARAGRAPH_B, PARAGRAPH_C],
        "expected_error": None,
    },
    "line_split_when_paragraph_too_long": {
        "description": "Falls back to line splitting when a paragraph exceeds chunk_size",
        "text": "\n".join([LINE_X, LINE_Y, LINE_Z]),
        "chunk_size": LINE_CHUNK_SIZE,
        "chunk_overlap": 5,
        "expected_chunks": [LINE_X, LINE_Y, LINE_Z],
        "expected_error": None,
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
        "expected_error": None,
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
        "expected_error": None,
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
        "expected_error": None,
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
        "expected_error": None,
    },
    "empty_string_returns_empty_list": {
        "description": "Empty input returns an empty chunk list",
        "text": "",
        "chunk_size": 100,
        "chunk_overlap": 10,
        "expected_chunks": [],
        "expected_error": None,
    },
    "no_chunk_is_ever_empty": {
        "description": "Blank paragraphs in the input never produce empty-string chunks",
        "text": "\n\n".join(["", WORD_A, "", WORD_B, ""]),
        "chunk_size": 20,
        "chunk_overlap": 2,
        "expected_chunks": [WORD_A, WORD_B],
        "expected_error": None,
    },
    "overlap_equals_size_raises": {
        "description": "chunk_overlap == chunk_size is invalid",
        "text": "some text",
        "chunk_size": 100,
        "chunk_overlap": 100,
        "expected_chunks": None,
        "expected_error": ValueError,
    },
    "overlap_greater_than_size_raises": {
        "description": "chunk_overlap > chunk_size is invalid",
        "text": "some text",
        "chunk_size": 100,
        "chunk_overlap": 150,
        "expected_chunks": None,
        "expected_error": ValueError,
    },
    "chunk_size_zero_raises": {
        "description": "chunk_size == 0 is invalid",
        "text": "some text",
        "chunk_size": 0,
        "chunk_overlap": 0,
        "expected_chunks": None,
        "expected_error": ValueError,
    },
    "chunk_size_negative_raises": {
        "description": "negative chunk_size is invalid",
        "text": "some text",
        "chunk_size": -10,
        "chunk_overlap": 0,
        "expected_chunks": None,
        "expected_error": ValueError,
    },
    "chunk_overlap_negative_raises": {
        "description": "negative chunk_overlap is invalid",
        "text": "some text",
        "chunk_size": 100,
        "chunk_overlap": -1,
        "expected_chunks": None,
        "expected_error": ValueError,
    },
}


@pytest.mark.parametrize("case", CASES.values(), ids=CASES.keys())
def test_recursive_chunk(case):
    if case["expected_error"] is not None:
        with pytest.raises(case["expected_error"]):
            recursive_chunk(
                text=case["text"],
                chunk_size=case["chunk_size"],
                chunk_overlap=case["chunk_overlap"],
            )
        return

    chunks = recursive_chunk(
        text=case["text"],
        chunk_size=case["chunk_size"],
        chunk_overlap=case["chunk_overlap"],
    )
    assert chunks == case["expected_chunks"], case["description"]
