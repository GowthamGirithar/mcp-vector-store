"""Factory for creating vector database adapters."""

import logging
from typing import Dict, Any, Type

from .base import BaseVectorDBAdapter
from .chroma import ChromaAdapter
from ..config.config import VectorDBConfig
from ..utils.exceptions import UnsupportedProviderError, ConfigurationError

logger = logging.getLogger(__name__)


class VectorDBFactory:
    """Factory for creating vector database adapters."""
    
    # Registry of available adapters
    _adapters: Dict[str, Type[BaseVectorDBAdapter]] = {
        "chroma": ChromaAdapter,
        # Future adapters can be added here:
        # "pinecone": PineconeAdapter,
        # "weaviate": WeaviateAdapter,
        # "qdrant": QdrantAdapter,
    }
    
    @classmethod
    def create_adapter(cls, config: VectorDBConfig) -> BaseVectorDBAdapter:
        """Create a vector database adapter based on configuration.
        
        Args:
            config: Vector database configuration
            
        Returns:
            Vector database adapter instance
            
        Raises:
            UnsupportedProviderError: If provider is not supported
            ConfigurationError: If configuration is invalid
        """
        provider = config.provider.lower()
        
        if provider not in cls._adapters:
            supported = list(cls._adapters.keys())
            raise UnsupportedProviderError(
                f"Unsupported vector database provider: {provider}. "
                f"Supported providers: {supported}"
            )
        
        adapter_class = cls._adapters[provider]
        
        try:
            adapter = adapter_class(config)
            logger.info(f"Created {provider} adapter")
            return adapter
        except Exception as e:
            logger.error(f"Failed to create {provider} adapter: {e}")
            raise ConfigurationError(f"Adapter creation failed: {e}")
    
    
    @classmethod
    def get_supported_providers(cls) -> list[str]:
        """Get list of supported vector database providers.
        
        Returns:
            List of provider names
        """
        return list(cls._adapters.keys())
    
    @classmethod
    def is_provider_supported(cls, provider: str) -> bool:
        """Check if a provider is supported.
        
        Args:
            provider: Provider name
            
        Returns:
            True if supported
        """
        return provider.lower() in cls._adapters
    