import asyncio
import logging
from typing import List
from sentence_transformers import SentenceTransformer

from .base import EmbeddingService

logger = logging.getLogger(__name__)

class SentenceTransformerEmbeddingService(EmbeddingService):
    """Handles local execution of SentenceTransformer models."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name_str = model_name
        self._model: SentenceTransformer = SentenceTransformer(self.model_name_str)
        self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    async def generate_embedding(self, text: str) -> List[float]:
        loop = asyncio.get_running_loop()
        embedding = await loop.run_in_executor(
            None, lambda: self._model.encode(text, convert_to_numpy=True)
        )
        return embedding.tolist()

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: self._model.encode(texts, convert_to_numpy=True)
        )
        return embeddings.tolist()

