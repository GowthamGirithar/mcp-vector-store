from typing import Optional
from .base import EmbeddingService
from .local_service import SentenceTransformerEmbeddingService

def create_embedding_service(
        provider: str,
        model: str,
        api_key: Optional[str] = None,
        enable_cache: bool = True,
        cache_size: int = 1000
) -> EmbeddingService:
    """Factory helper to instantiate any service easily."""
    provider_clean = provider.lower().strip()
    if provider_clean in ["sentence_transformers", "local"]:
        service = SentenceTransformerEmbeddingService(model_name=model)
    else:
        raise ValueError(f"Unsupported provider: '{provider}'")

    return service