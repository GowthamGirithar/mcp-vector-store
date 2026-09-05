"""Boot real vector-db + embedding services outside the FastMCP lifespan.

`generate_document_embedding`/`hybrid_search` read their services through
`mcp_vectordb.services.get_vector_db()`/`get_embedding_service()`, which are
only populated inside the FastMCP server's lifespan (`services.setup_services`).
Eval scripts call the tool functions directly without running that server, so
this installs the same globals `tests/conftest.py`'s `real_services` fixture
does, pointed at a dedicated on-disk Chroma path so eval runs don't touch the
project's real `chroma_db` directory.
"""

import mcp_vectordb.services as services_module
from mcp_vectordb.adapters.factory import VectorDBFactory
from mcp_vectordb.config.config import get_settings
from mcp_vectordb.embedding import create_embedding_service

EVAL_CHROMA_PATH = "tests/eval/.eval_chroma_db"


async def init_eval_services():
    """Install real vector-db + local embedding services as the `services`
    module globals, and return them. Idempotent: safe to call once per
    process before any tool call."""
    settings = get_settings()
    settings.vector_db.path = EVAL_CHROMA_PATH

    vector_db = VectorDBFactory.create_adapter(settings.vector_db)
    await vector_db.initialize()

    embedding_service = create_embedding_service(
        provider="sentence-transformers",
        model=settings.embedding.model
        if settings.embedding.provider in ("sentence-transformers", "sentence_transformers", "local")
        else "all-MiniLM-L6-v2",
        enable_cache=True,
    )

    services_module.vector_db = vector_db
    services_module.embedding_service = embedding_service
    return vector_db, embedding_service
