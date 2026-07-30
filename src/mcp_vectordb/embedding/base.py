from abc import ABC, abstractmethod
from typing import List

class EmbeddingService(ABC):
    """Abstract interface for all embedding services."""

    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        pass

    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        pass
