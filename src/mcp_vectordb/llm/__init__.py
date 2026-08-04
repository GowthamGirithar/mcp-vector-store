from typing import Optional
from .base import LLMService
from .openai_service import OpenAILLMService


def create_llm_service(
    provider: str,
    model: str,
    api_key: Optional[str] = None,
) -> LLMService:
    """Factory helper to instantiate an LLMService for the configured provider."""
    provider_clean = provider.lower().strip()
    if provider_clean == "openai":
        return OpenAILLMService(model=model, api_key=api_key)

    raise ValueError(f"Unsupported LLM provider: '{provider}'")
