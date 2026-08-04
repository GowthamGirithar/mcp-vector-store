from abc import ABC, abstractmethod


class LLMService(ABC):
    """Abstract interface for LLM completion services."""

    @abstractmethod
    async def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Run a chat completion and return the raw text response."""
        pass
