"""Service initialization and lifecycle management."""

import logging
from contextlib import asynccontextmanager

from .config.config import get_settings
from .core.embedding import EmbeddingServiceFactory, CachedEmbeddingService
from .adapters.factory import VectorDBFactory

logger = logging.getLogger(__name__)

# Global services
vector_db = None
embedding_service = None
settings = None


@asynccontextmanager
#asynchronous context manager using async def and yield.
# we can use asynccontextmanager when we need to setup before and clean up after - using async and yield
async def setup_services(app):
    print("🚀 setup_services called!")
    """Initialize and cleanup services for FastMCP server."""
    global vector_db, embedding_service, settings
    
    try:
        logger.info("Initializing MCP Vector Database Server")
        
        # Load settings
        settings = get_settings()
        logger.info(f"Loaded configuration for provider: {settings.vector_db.provider}")
        
        # Initialize embedding service
        embedding_service_impl = EmbeddingServiceFactory.create_service(settings.embedding)
        embedding_service = CachedEmbeddingService(embedding_service_impl, cache_size=1000)
        logger.info(f"Initialized embedding service: {settings.embedding.provider}")
        
        # Initialize vector database adapter
        vector_db = VectorDBFactory.create_adapter(settings.vector_db)
        await vector_db.initialize()
        logger.info(f"Initialized vector database: {settings.vector_db.provider}")
        
        # Health check
        if await vector_db.health_check():
            logger.info("Vector database health check passed")
        else:
            logger.warning("Vector database health check failed")
        
        logger.info("MCP Vector Database Server initialization complete")
        
        yield  # Server runs here
        
    except Exception as e:
        logger.error(f"Failed to initialize server: {e}")
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up MCP Vector Database Server")
        
        if vector_db:
            await vector_db.close()
            logger.info("Closed vector database connection")
        
        if embedding_service and hasattr(embedding_service, 'cache'):
            embedding_service.cache.clear()
            logger.info("Cleared embedding cache")
        
        logger.info("Server cleanup complete")


def get_vector_db():
    """Get the global vector database instance."""
    return vector_db


def get_embedding_service():
    """Get the global embedding service instance."""
    return embedding_service


def get_settings_instance():
    """Get the global settings instance."""
    return settings