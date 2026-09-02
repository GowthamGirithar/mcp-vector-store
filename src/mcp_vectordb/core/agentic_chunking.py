"""Agentic chunking: LLM-driven chunk boundaries and record shape for the
agentic-embedding tool.

Unlike the fixed-size/title-based chunker in `chunking/process_document.py`,
here the LLM decides both where chunk boundaries fall and the entire output
record shape, in a single pass over the full text — driven entirely by the
caller-supplied `prompt`. The prompt may ask for a bare JSON array of
arbitrary objects (no `text` field at all, e.g. structured extraction like
question/answer pairs) or for chunk objects carrying a verbatim `text` field
plus tags. Both are supported: if a record has a `text` field, that's what
gets embedded and the rest becomes metadata; otherwise the whole record is
serialized and embedded, with all its fields kept as metadata.
"""

import json
import logging
from typing import Any, Dict, List, NamedTuple

from ..chunking.raw_text import extract_raw_text, extract_title_sections
from ..llm.base import LLMService
from ..utils.exceptions import LLMServiceError

try:
    from langsmith import traceable
except ImportError:
    def traceable(*_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are given a task and a document's text. Follow the task exactly, "
    "including any JSON structure it specifies.\n\n"
    "Respond with ONLY valid JSON — either a JSON array or a JSON object, "
    "in whatever shape the task asks for. No prose, no markdown code "
    "fences, no explanation before or after the JSON."
)


class AgenticChunk(NamedTuple):
    """One LLM-produced record: the text to embed plus its full field set as metadata."""

    text: str
    metadata: Dict[str, Any]


def _strip_code_fence(raw: str) -> str:
    text = raw.strip()
    if not text.startswith("```"):
        return text
    text = text[3:]
    if text.lower().startswith("json"):
        text = text[4:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


def _parse_json_response(raw: str) -> Any:
    try:
        return json.loads(_strip_code_fence(raw))
    except (TypeError, ValueError) as exc:
        raise LLMServiceError(f"LLM did not return valid JSON: {exc}") from exc


def _extract_records(parsed: Any) -> List[Dict[str, Any]]:
    if isinstance(parsed, list):
        return [record for record in parsed if isinstance(record, dict)]

    if isinstance(parsed, dict):
        for value in parsed.values():
            if isinstance(value, list):
                return [record for record in value if isinstance(record, dict)]
        # No array found anywhere in the object — treat it as one record.
        return [parsed]

    raise LLMServiceError(
        f"LLM JSON response was neither an array nor an object (got {type(parsed).__name__})"
    )


@traceable(run_type="chain", name="agentic_chunk_text")
async def agentic_chunk_text(
    text: str, prompt: str, llm_service: LLMService, max_input_chars: int
) -> List[AgenticChunk]:
    """Run `prompt` against `text` with the LLM and turn the result into chunks.

    Sends `text` to the LLM in a single call. `max_input_chars` here is a
    last-resort safety net (truncates and logs a warning if exceeded) — the
    caller is expected to have already kept `text` under that size, e.g. via
    `agentic_chunk_document`'s title-split fallback for file input.

    Args:
        text: The text to process in one LLM call.
        prompt: Caller instructions — fully controls chunk boundaries and the
            output record shape.
        llm_service: The LLM completion service to call.
        max_input_chars: Safety-net cap on `text` size sent to the LLM (see
            `LLMConfig.max_input_chars`).

    Returns:
        One chunk per record the LLM returned.

    Raises:
        LLMServiceError: If the LLM call fails or its response isn't
            parseable JSON.
    """
    if len(text) > max_input_chars:
        logger.warning(
            "agentic_chunk_text: truncating text from %d to %d chars (max_input_chars)",
            len(text), max_input_chars,
        )
        text = text[:max_input_chars]

    user_prompt = f"{prompt}\n\nDocument text:\n{text}"
    raw_response = await llm_service.complete(_SYSTEM_PROMPT, user_prompt)
    records = _extract_records(_parse_json_response(raw_response))

    chunks: List[AgenticChunk] = []
    for record in records:
        record_text = record.get("text")
        if isinstance(record_text, str) and record_text.strip():
            chunk_text = record_text.strip()
            chunk_metadata = {key: value for key, value in record.items() if key != "text"}
        else:
            chunk_text = json.dumps(record, ensure_ascii=False)
            chunk_metadata = record

        chunks.append(AgenticChunk(text=chunk_text, metadata=chunk_metadata))

    return chunks


@traceable(run_type="chain", name="agentic_chunk_document")
async def agentic_chunk_document(
    file_path: str, prompt: str, llm_service: LLMService, max_input_chars: int
) -> List[AgenticChunk]:
    """Run `prompt` against a document with the LLM, splitting only if needed.

    Extracts the document's full raw text. If it fits under
    `max_input_chars`, it's sent to the LLM in one call — preserving full
    document context, which matters for tasks that need to reason across
    the whole document (e.g. matching a question to an answer key in a
    different section).

    Only when the document is too large for one call does this fall back to
    splitting on title boundaries (`extract_title_sections`) and running one
    LLM call per section, concatenating the results. That fallback trades
    away cross-section correctness for the ability to handle arbitrarily
    large documents at all.

    Args:
        file_path: Path to the document to process.
        prompt: Caller instructions — fully controls chunk boundaries and the
            output record shape.
        llm_service: The LLM completion service to call.
        max_input_chars: Max characters sent to the LLM in one call (see
            `LLMConfig.max_input_chars`).

    Returns:
        One chunk per record the LLM returned, across all sections.
    """
    full_text = extract_raw_text(file_path)
    if len(full_text) <= max_input_chars:
        return await agentic_chunk_text(full_text, prompt, llm_service, max_input_chars)

    logger.warning(
        "agentic_chunk_document: '%s' (%d chars) exceeds max_input_chars=%d; "
        "falling back to per-title-section LLM calls. Tasks that need to "
        "reason across sections (e.g. matching a question to an answer key "
        "in a different section) are not guaranteed correct in this mode.",
        file_path, len(full_text), max_input_chars,
    )

    sections = extract_title_sections(file_path, max_input_chars)
    chunks: List[AgenticChunk] = []
    for section in sections:
        chunks.extend(await agentic_chunk_text(section, prompt, llm_service, max_input_chars))
    return chunks
