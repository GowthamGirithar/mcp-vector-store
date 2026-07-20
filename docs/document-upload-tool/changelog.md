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
