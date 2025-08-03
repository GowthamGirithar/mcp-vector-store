"""Custom exceptions for MCP Vector DB Server."""


class VectorDBError(Exception):
    """Base exception for vector database operations."""
    
    def __init__(self, message: str, details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class EmbeddingError(VectorDBError):
    """Exception raised when embedding generation fails."""
    pass


class ConfigurationError(VectorDBError):
    """Exception raised when configuration is invalid."""
    pass


class ValidationError(VectorDBError):
    """Exception raised when input validation fails."""
    pass


class ConnectionError(VectorDBError):
    """Exception raised when connection to vector database fails."""
    pass


class CollectionError(VectorDBError):
    """Exception raised when collection operations fail."""
    pass


class DocumentNotFoundError(VectorDBError):
    """Exception raised when a document is not found."""
    pass


class UnsupportedProviderError(ConfigurationError):
    """Exception raised when an unsupported provider is specified."""
    pass


class EmbeddingModelError(EmbeddingError):
    """Exception raised when embedding model operations fail."""
    pass


class SearchError(VectorDBError):
    """Exception raised when similarity search fails."""
    pass


class StorageError(VectorDBError):
    """Exception raised when document storage fails."""
    pass