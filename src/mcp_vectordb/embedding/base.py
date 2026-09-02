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

    @property
    @abstractmethod
    def max_input_tokens(self) -> int:
        """Maximum number of tokens the underlying model accepts per input.

        Text beyond this length is silently truncated by the model, not by
        this service — callers that need to avoid truncation (chunking,
        ingestion reporting) should size their input using `count_tokens`
        against this budget before calling `generate_embedding(s)`.
        """
        pass

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Count how many tokens `text` will occupy for this model.

        Uses the same tokenizer the model itself will apply, so the result
        matches what `max_input_tokens` is measured in.
        """
        pass
