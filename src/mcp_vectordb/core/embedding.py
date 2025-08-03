"""Embedding service implementations for MCP Vector DB Server."""

import asyncio
import logging
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

import openai
import numpy as np
from sentence_transformers import SentenceTransformer

from .interfaces import EmbeddingService
from ..utils.exceptions import EmbeddingError, EmbeddingModelError, ConfigurationError
from ..config.config import EmbeddingConfig

logger = logging.getLogger(__name__)


class OpenAIEmbeddingService(EmbeddingService):
    """OpenAI embedding service implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize OpenAI embedding service.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.model = config.model
        self._dimension = None
        
        if not config.api_key:
            raise ConfigurationError("OpenAI API key is required")
        
        # Initialize OpenAI client
        openai.api_key = config.api_key
        self.client = openai.OpenAI(api_key=config.api_key)
        
        # Model dimension mapping
        # TODO: may be load on run time depends on the model specified
        self._model_dimensions = {
            "text-embedding-ada-002": 1536,
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
        }
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
            )
            # open ai returns just python list as result
            # With local models like sentence-transformers, the result is a NumPy array or tensor, because you're using PyTorch or NumPy under the hood.
            embedding = response.data[0].embedding
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embedding: {e}")
            raise EmbeddingError(f"OpenAI embedding generation failed: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            # OpenAI supports batch processing
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self.client.embeddings.create(
                    input=texts,
                    model=self.model
                )
            )
            
            embeddings = [item.embedding for item in response.data]
            logger.debug(f"Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate OpenAI embeddings: {e}")
            raise EmbeddingError(f"OpenAI batch embedding generation failed: {e}")
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if self._dimension is None:
            self._dimension = self._model_dimensions.get(self.model, 1536)
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        return self.model


class SentenceTransformerEmbeddingService(EmbeddingService):
    """Sentence Transformers embedding service implementation."""
    
    def __init__(self, config: EmbeddingConfig):
        """Initialize Sentence Transformers embedding service.
        
        Args:
            config: Embedding configuration
        """
        self.config = config
        self.model_name_str = config.model
        self._model = None
        self._dimension = None
        
        # Default model if not specified
        if not self.model_name_str:
            self.model_name_str = "all-MiniLM-L6-v2"
    
    def _load_model(self) -> SentenceTransformer:
        """Load the sentence transformer model."""
        if self._model is None:
            try:
                self._model = SentenceTransformer(self.model_name_str)
            except Exception as e:
                raise EmbeddingModelError(f"Failed to load model {self.model_name_str}: {e}")
        
        return self._model
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            model = self._load_model()
            
            # Run in thread pool to avoid blocking
            # otherwise Other async tasks have to wait (no concurrency).
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: model.encode(text, convert_to_numpy=True) # we can tensor when it is deep learning
            )
            # dimention of embedding for each model is fixed and we cannot update it.
            # also why we have collection dimention if this is fixed 
            # - Distance or similarity computations would be invalid or nonsensical.
            # Text is divided into chunks
            # Each chunk produce the one embeddings
            # each models has embeddings of different dimentions
            # we can insert into collection only if our embeddings dimention matches with collection dimensions
            
            # Convert numpy array to list
            embedding_list = embedding.tolist()
            logger.debug(f"Generated embedding for text of length {len(text)}")
            return embedding_list
            
        except Exception as e:
            logger.error(f"Failed to generate SentenceTransformer embedding: {e}")
            raise EmbeddingError(f"SentenceTransformer embedding generation failed: {e}")
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
            
        Raises:
            EmbeddingError: If embedding generation fails
        """
        try:
            model = self._load_model()
            
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(texts, convert_to_numpy=True)
            )
            # dimention of embedding for each model is fixed and we cannot update it.
            # also why we have collection dimention if this is fixed 
            # - Distance or similarity computations would be invalid or nonsensical.
            # Text is divided into chunks
            # Each chunk produce the one embeddings
            # each models has embeddings of different dimentions
            # we can insert into collection only if our embeddings dimention matches with collection dimensions
            
            # Convert numpy array to list of lists
            embeddings_list = embeddings.tolist()
            logger.debug(f"Generated {len(embeddings_list)} embeddings")
            return embeddings_list
            
        except Exception as e:
            raise EmbeddingError(f"SentenceTransformer batch embedding generation failed: {e}")
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        if self._dimension is None:
            model = self._load_model()
            self._dimension = model.get_sentence_embedding_dimension()
        return self._dimension
    
    @property
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        return self.model_name_str


class EmbeddingServiceFactory:
    """Factory for creating embedding services."""
    
    @staticmethod
    def create_service(config: EmbeddingConfig) -> EmbeddingService:
        """Create an embedding service based on configuration.
        
        Args:
            config: Embedding configuration
            
        Returns:
            Embedding service instance
            
        Raises:
            ConfigurationError: If provider is not supported
        """
        provider = config.provider.lower()
        
        if provider == "openai":
            return OpenAIEmbeddingService(config)
        elif provider == "sentence_transformers":
            return SentenceTransformerEmbeddingService(config)
        else:
            raise ConfigurationError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def get_supported_providers() -> List[str]:
        """Get list of supported embedding providers.
        
        Returns:
            List of provider names
        """
        return ["openai", "sentence_transformers"]


class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    # An embedding cache stores previously generated embeddings, so you don’t re-embed the same text again.
    # Instead of recomputing, you reuse the cached embedding — which is faster and cheaper.
    
    def __init__(self, max_size: int = 1000):
        """Initialize embedding cache.
        
        Args:
            max_size: Maximum number of cached embeddings
        """
        self.max_size = max_size
        self._cache: Dict[str, List[float]] = {}
        self._access_order: List[str] = []
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get cached embedding for text.
        
        Args:
            text: The text to look up
            
        Returns:
            Cached embedding or None
        """
        if text in self._cache:
            # Move to end (most recently used)
            self._access_order.remove(text)
            self._access_order.append(text)
            return self._cache[text]
        return None
    
    def put(self, text: str, embedding: List[float]) -> None:
        """Cache an embedding.
        
        Args:
            text: The text
            embedding: The embedding vector
        """
        if text in self._cache:
            # Update existing
            self._cache[text] = embedding
            self._access_order.remove(text)
            self._access_order.append(text)
        else:
            # Add new
            if len(self._cache) >= self.max_size:
                # Remove least recently used
                oldest = self._access_order.pop(0)
                del self._cache[oldest]
            
            self._cache[text] = embedding
            self._access_order.append(text)
    
    def clear(self) -> None:
        """Clear the cache."""
        self._cache.clear()
        self._access_order.clear()
    
    def size(self) -> int:
        """Get cache size."""
        return len(self._cache)


class CachedEmbeddingService(EmbeddingService):
    """Wrapper that adds caching to any embedding service."""
    
    def __init__(self, service: EmbeddingService, cache_size: int = 1000):
        """Initialize cached embedding service.
        
        Args:
            service: The underlying embedding service
            cache_size: Maximum cache size
        """
        self.service = service
        self.cache = EmbeddingCache(cache_size)
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with caching.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        # Check cache first
        cached = self.cache.get(text)
        if cached is not None:
            logger.debug(f"Cache hit for text of length {len(text)}")
            return cached
        
        # Generate and cache
        embedding = await self.service.generate_embedding(text)
        self.cache.put(text, embedding)
        logger.debug(f"Cache miss, generated and cached embedding for text of length {len(text)}")
        return embedding
    
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings with caching.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        uncached_texts = []
        uncached_indices = []
        
        # Check cache for each text
        for i, text in enumerate(texts):
            cached = self.cache.get(text)
            if cached is not None:
                embeddings.append(cached)
            else:
                embeddings.append(None)  # Placeholder
                uncached_texts.append(text)
                uncached_indices.append(i)
        
        # Generate embeddings for uncached texts
        if uncached_texts:
            new_embeddings = await self.service.generate_embeddings(uncached_texts)
            
            # Cache and insert new embeddings
            # zip combines into pairs
            for idx, embedding in zip(uncached_indices, new_embeddings):
                self.cache.put(texts[idx], embedding)
                embeddings[idx] = embedding
        
        logger.debug(f"Generated {len(uncached_texts)} new embeddings, {len(texts) - len(uncached_texts)} from cache")
        return embeddings
    
    @property
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        return self.service.dimension
    
    @property
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        return self.service.model_name