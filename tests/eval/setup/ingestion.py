"""Per-run ingestion setup for the eval: idempotently ensure the fixture PDF
is ingested, then build the join key back to ground truth.
"""

from typing import Dict

from mcp_vectordb.services import get_vector_db
from mcp_vectordb.tools.document_embedding import generate_document_embedding


async def ensure_ingested(file_path: str, collection: str) -> None:
    """Ingest `file_path` into `collection` if not already present.

    force=False: a prior ingestion (matched by content hash) is left as-is,
    so chunk_ids/chunk_index assignments stay stable across eval runs rather
    than churning on every invocation.
    """
    result = await generate_document_embedding(
        file_path=file_path, collection=collection, force=False, ctx=None
    )
    print(result)


async def map_text_to_chunk_index(collection: str) -> Dict[str, int]:
    """Map each stored chunk's exact text to its chunk_index.

    hybrid_search returns bare text strings, not ids, and chunk_ids are
    regenerated on every ingestion — chunk_index (deterministic parse order)
    is the only stable key the golden dataset can reference, so results are
    joined back to ground truth via an exact text match against this map.
    """
    vector_db = get_vector_db()
    docs = await vector_db.get_all_documents(collection)
    return {doc.text: doc.metadata.get("chunk_index") for doc in docs}
