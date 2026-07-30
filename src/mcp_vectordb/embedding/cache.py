import logging
from typing import Dict, List, Optional
from .base import EmbeddingService

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """LRU-style in-memory cache for text embeddings."""

    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []

    def get(self, text: str) -> Optional[List[float]]:
        if text in self._cache:
            self._access_order.remove(text)
            self._access_order.append(text)
            return self._cache[text]
        return None

    def put(self, text: str, embedding: List[float]) -> None:
        if text in self._cache:
            self._access_order.remove(text)
        elif len(self._cache) >= self.max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]

        self._cache[text] = embedding
        self._access_order.append(text)

    def clear(self) -> None:
        self._cache.clear()
        self._access_order.clear()


class CachedEmbeddingService(EmbeddingService):
    """Decorator adding cache support to any EmbeddingService."""

    def __init__(self, service: EmbeddingService, cache_size: int = 1000):
        self.service = service
        self.cache = EmbeddingCache(cache_size)

    @property
    def dimension(self) -> int:
        return self.service.dimension

    async def generate_embedding(self, text: str) -> List[float]:
        if cached := self.cache.get(text):
            return cached
        embedding = await self.service.generate_embedding(text)
        self.cache.put(text, embedding)
        return embedding

    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        embeddings, uncached_texts, uncached_indices = [], [], []

        for idx, text in enumerate(texts):
            if cached := self.cache.get(text):
                embeddings.append(cached)
            else:
                embeddings.append(None)
                uncached_texts.append(text)
                uncached_indices.append(idx)

        if uncached_texts:
            new_embeddings = await self.service.generate_embeddings(uncached_texts)
            for idx, emb in zip(uncached_indices, new_embeddings):
                self.cache.put(texts[idx], emb)
                embeddings[idx] = emb

        return embeddings
