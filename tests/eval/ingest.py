"""One-off: ingest tests/fixtures/attention.pdf into the fixed eval
collection and dump every stored chunk (id + text) to a JSON file, so the
golden dataset (tests/eval/golden/attention_qa.jsonl) can be hand-written
against real chunk ids/boundaries instead of guessed ones.

Usage: .venv/bin/python -m tests.eval.ingest
"""

import asyncio
import json

from tests.eval.setup_services import init_eval_services
from mcp_vectordb.services import get_vector_db
from mcp_vectordb.tools.document_embedding import generate_document_embedding

EVAL_COLLECTION = "eval_attention"
PDF_PATH = "tests/fixtures/attention.pdf"
DUMP_PATH = "tests/eval/golden/attention_chunks.json"


async def main():
    await init_eval_services()

    result = await generate_document_embedding(
        file_path=PDF_PATH,
        collection=EVAL_COLLECTION,
        force=True,
        ctx=None,
    )
    print(result)

    vector_db = get_vector_db()
    chunks = await vector_db.get_all_documents(EVAL_COLLECTION)
    dump = [
        {
            "chunk_id": c.id,
            "chunk_index": c.metadata.get("chunk_index"),
            "has_table": c.metadata.get("has_table"),
            "has_image": c.metadata.get("has_image"),
            "text": c.text,
        }
        for c in sorted(chunks, key=lambda d: d.metadata.get("chunk_index", 0))
    ]
    with open(DUMP_PATH, "w") as f:
        json.dump(dump, f, indent=2)
    print(f"Dumped {len(dump)} chunks to {DUMP_PATH}")


if __name__ == "__main__":
    asyncio.run(main())
