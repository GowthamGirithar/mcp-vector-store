# document-upload-tool changelog

## T2: Recursive chunking

- Implemented `recursive_chunk(text, chunk_size, chunk_overlap)` in
  `src/mcp_vectordb/core/chunking.py`. Tries paragraph (`\n\n`) splits
  first, then line (`\n`) splits, then sentence (`". "`) splits, then
  falls back to a hard character window with overlap for any piece that
  is still too long after all separator-based splits.
- Test setup: cut from `main`, which does not yet have `tests/` or
  `pytest.ini` (T1's branch with those hasn't merged yet). Created a
  root `pytest.ini` with `pythonpath = src` / `testpaths = tests` to
  match T1's convention, and added `tests/test_chunking.py`. No
  `conftest.py` was needed since `recursive_chunk` is a pure function
  with no fixtures/config dependencies.
- Empty string input returns `[]` (documented and asserted behavior).
- `ValueError` is raised for `chunk_size <= 0`, `chunk_overlap < 0`, and
  `chunk_overlap >= chunk_size`.
- The hard-window fallback advances by `chunk_size - chunk_overlap`
  characters per step, which is always >= 1 given the validated inputs,
  guaranteeing termination even on pathological input with no
  separators at all (verified with a 100k-char single-token string in
  tests).
- Overlap correctness is verified by asserting the tail of chunk[i]
  (last `chunk_overlap` chars) appears at the start of chunk[i+1] in the
  hard-window fallback path.

## T3: Text extraction (PDF/TXT/MD)

- Implemented `extract_text(file_path) -> List[Tuple[Optional[int], str]]`
  in `src/mcp_vectordb/core/parsers.py`. Dispatches on the lowercased file
  extension: `.pdf` returns one `(page_number, text)` tuple per page
  (1-indexed) via `pypdf.PdfReader`; `.txt`/`.md` return a single
  `(None, full_text)` tuple read as UTF-8. Any other extension raises
  `UnsupportedFileTypeError` (a `ValueError` subclass) before the file is
  opened.
- Added `DocumentParseError` (plain `Exception` subclass) which wraps
  `pypdf.errors.PdfReadError` (and any other unexpected exception from
  `pypdf`) so a corrupt/truncated PDF raises a distinct, documented
  exception instead of an unrelated traceback. Note for T5 (or whichever
  task maps this to the tool layer): catch `DocumentParseError` and map it
  to `RuntimeError`; catch `UnsupportedFileTypeError` separately if a
  different mapping (e.g. a 4xx-style tool error) is wanted, since it is
  intentionally a `ValueError` subtype, not a `DocumentParseError`.
- Added `pypdf>=4.0.0` to `requirements.txt`.
- Test fixtures added under `tests/fixtures/`: `sample.txt`, `sample.md`,
  `sample.docx` (empty file, only used to exercise the
  extension-dispatch failure path before any parsing is attempted),
  `sample.pdf` (valid 3-page PDF with extractable text per page), and
  `corrupt.pdf` (a truncated copy of `sample.pdf`, cut to roughly a third
  of its byte length, which reliably raises `pypdf.errors.PdfStreamError`
  at `PdfReader(...)` construction time).
- PDF fixture generation approach: installed `reportlab` as a dev-only
  helper (not added to `requirements-dev.txt`, only used locally to
  generate fixtures) and used `reportlab.pdfgen.canvas` to draw real text
  on 3 pages, then saved. This was simpler and more robust than
  hand-crafting PDF bytes or trying to get extractable text out of a
  `pypdf.PdfWriter`-only blank-page PDF (blank pages have no text to
  extract, which would fail the "non-empty extracted text per page"
  assertion). The generation script is not committed (one-off, run from
  the project venv); only the resulting fixture files are checked in.
- Test setup: cut from `main` (no `tests/`/`pytest.ini` there yet, same as
  T1/T2). Reused the same root `pytest.ini` convention
  (`pythonpath = src`, `testpaths = tests`) — no changes needed since T1's
  version isn't merged yet and this branch created its own copy.
  Installed `pypdf` and `reportlab` into the existing project `.venv`.
