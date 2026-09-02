import asyncio
import logging
from typing import List
from sentence_transformers import SentenceTransformer

from .base import EmbeddingService

try:
    from langsmith import traceable
    from langsmith.run_helpers import get_current_run_tree
except ImportError:
    def traceable(*_args, **_kwargs):
        def _decorator(fn):
            return fn
        return _decorator

    def get_current_run_tree():
        return None

logger = logging.getLogger(__name__)


def _embedding_output(embedding: List[float]) -> dict:
    # Trace the vector's shape, not its (large, unreadable) contents.
    return {"dimension": len(embedding)}


def _embeddings_output(embeddings: List[List[float]]) -> dict:
    return {"count": len(embeddings), "dimension": len(embeddings[0]) if embeddings else 0}


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Handles local execution of SentenceTransformer models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name_str = model_name
        self._model: SentenceTransformer = SentenceTransformer(self.model_name_str)
        self._dimension = self._model.get_embedding_dimension()
        self._truncated_count = 0

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def max_input_tokens(self) -> int:
        return self._model.max_seq_length

    def count_tokens(self, text: str) -> int:
        return len(self._model.tokenizer.encode(text, add_special_tokens=True))

    def _warn_if_truncated(self, text: str) -> None:
        n = self.count_tokens(text)
        limit = self.max_input_tokens
        if n > limit:
            self._truncated_count += 1
            logger.warning(
                "Embedding input truncated: %d tokens > model limit %d (%.0f%% discarded, "
                "total truncated so far: %d). Chunk sizing should keep inputs under this "
                "limit — see chunking/token_budget.py.",
                n, limit, 100 * (1 - limit / n), self._truncated_count,
            )

    def _tag_current_run(self) -> None:
        if (run_tree := get_current_run_tree()) is not None:
            run_tree.add_metadata({
                "ls_provider": "sentence-transformers",
                "ls_model_name": self.model_name_str,
                "dimension": self._dimension,
            })

    @traceable(
        run_type="embedding",
        name="sentence_transformer.generate_embedding",
        process_outputs=_embedding_output,
    )
    async def generate_embedding(self, text: str) -> List[float]:
        self._warn_if_truncated(text)
        self._tag_current_run()
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self._model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()

    @traceable(
        run_type="embedding",
        name="sentence_transformer.generate_embeddings",
        process_outputs=_embeddings_output,
    )
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        for text in texts:
            self._warn_if_truncated(text)
        self._tag_current_run()
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, convert_to_numpy=True)
        )
        return embeddings.tolist()

