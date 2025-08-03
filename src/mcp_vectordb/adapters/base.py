"""Base adapter implementation for vector databases."""

import logging
from abc import ABC
from typing import List, Dict, Any, Optional
from ..core.document import Document, SearchResult
from ..core.interfaces import VectorDBAdapter
from ..utils.exceptions import VectorDBError, ConnectionError

logger = logging.getLogger(__name__)


class BaseVectorDBAdapter(VectorDBAdapter):
    """Base implementation for vector database adapters."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize base adapter.
        
        Args:
            config: Adapter configuration
        """
        self.config = config
        self._initialized = False
        self._connected = False
    
    async def initialize(self) -> None:
        """Initialize the vector database connection."""
        if self._initialized:
            return
        
        try:
            await self._connect()
            self._initialized = True
            self._connected = True
        except Exception as e:
            raise ConnectionError(f"Initialization failed: {e}")
    
    async def close(self) -> None:
        """Close the vector database connection."""
        if not self._connected:
            return
        
        try:
            await self._disconnect()
            self._connected = False
        except Exception as e:
            raise VectorDBError(f"Close failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if the vector database is healthy."""
        if not self._connected:
            return False
        
        try:
            return await self._health_check_impl()
        except Exception as e:
            return False
    
    def _validate_initialized(self) -> None:
        """Validate that the adapter is initialized."""
        if not self._initialized:
            raise VectorDBError("Adapter not initialized. Call initialize() first.")
    
    def _validate_connected(self) -> None:
        """Validate that the adapter is connected."""
        if not self._connected:
            raise ConnectionError("Adapter not connected to database.")
    
    def _validate_collection_name(self, collection: str) -> None:
        """Validate collection name."""
        if not collection or not isinstance(collection, str):
            raise VectorDBError("Collection name must be a non-empty string")
    
    def _validate_documents(self, documents: List[Document]) -> None:
        """Validate documents list."""
        if not documents:
            raise VectorDBError("Documents list cannot be empty")
        
        for doc in documents:
            if not isinstance(doc, Document):
                raise VectorDBError("All items must be Document instances")
            
            if not doc.text:
                raise VectorDBError(f"Document {doc.id} has empty text")
            
            if not doc.embedding:
                raise VectorDBError(f"Document {doc.id} has no embedding")
    
    def _validate_embedding(self, embedding: List[float]) -> None:
        """Validate embedding vector."""
        if not embedding:
            raise VectorDBError("Embedding cannot be empty")
        
        if not all(isinstance(x, (int, float)) for x in embedding):
            raise VectorDBError("Embedding must contain only numbers")
    
    def _validate_top_k(self, top_k: int) -> None:
        """Validate top_k parameter."""
        if not isinstance(top_k, int) or top_k <= 0:
            raise VectorDBError("top_k must be a positive integer")
        
        if top_k > 1000:  # Reasonable limit
            raise VectorDBError("top_k cannot exceed 1000")
    
    # Abstract methods that subclasses must implement
    async def _connect(self) -> None:
        """Connect to the vector database."""
        raise NotImplementedError
    
    async def _disconnect(self) -> None:
        """Disconnect from the vector database."""
        raise NotImplementedError
    
    async def _health_check_impl(self) -> bool:
        """Implementation-specific health check."""
        raise NotImplementedError
    
    # Default implementations that can be overridden
    async def create_collection(self, name: str, dimension: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection."""
        self._validate_initialized()
        self._validate_connected()
        self._validate_collection_name(name)
        
        if dimension <= 0:
            raise VectorDBError("Dimension must be positive")
        
        return await self._create_collection_impl(name, dimension, metadata)
    
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection."""
        self._validate_initialized()
        self._validate_connected()
        self._validate_collection_name(name)
        
        return await self._delete_collection_impl(name)
    
    async def list_collections(self) -> List[str]:
        """List all collections."""
        self._validate_initialized()
        self._validate_connected()
        
        return await self._list_collections_impl()
    
    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists."""
        self._validate_initialized()
        self._validate_connected()
        self._validate_collection_name(name)
        
        return await self._collection_exists_impl(name)
    
    async def store_documents(self, documents: List[Document], collection: str) -> List[str]:
        """Store documents in the vector database."""
        self._validate_initialized()
        self._validate_connected()
        self._validate_collection_name(collection)
        self._validate_documents(documents)
        
        return await self._store_documents_impl(documents, collection)
    
    async def similarity_search(
        self,
        query_embedding: List[float],
        collection: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search."""
        self._validate_initialized()
        self._validate_connected()
        self._validate_collection_name(collection)
        self._validate_embedding(query_embedding)
        self._validate_top_k(top_k)
        
        return await self._similarity_search_impl(query_embedding, collection, top_k, filters)
    
    async def get_document(self, doc_id: str, collection: str) -> Optional[Document]:
        """Get a document by ID."""
        self._validate_initialized()
        self._validate_connected()
        self._validate_collection_name(collection)
        
        if not doc_id:
            raise VectorDBError("Document ID cannot be empty")
        
        return await self._get_document_impl(doc_id, collection)
    
    async def delete_documents(self, doc_ids: List[str], collection: str) -> bool:
        """Delete documents by IDs."""
        self._validate_initialized()
        self._validate_connected()
        self._validate_collection_name(collection)
        
        if not doc_ids:
            raise VectorDBError("Document IDs list cannot be empty")
        
        return await self._delete_documents_impl(doc_ids, collection)
    
    async def update_document(self, doc_id: str, document: Document, collection: str) -> bool:
        """Update a document."""
        self._validate_initialized()
        self._validate_connected()
        self._validate_collection_name(collection)
        
        if not doc_id:
            raise VectorDBError("Document ID cannot be empty")
        
        if not isinstance(document, Document):
            raise VectorDBError("Document must be a Document instance")
        
        return await self._update_document_impl(doc_id, document, collection)
    
    async def count_documents(self, collection: str) -> int:
        """Count documents in a collection."""
        self._validate_initialized()
        self._validate_connected()
        self._validate_collection_name(collection)
        
        return await self._count_documents_impl(collection)
    
    # Abstract implementation methods
    async def _create_collection_impl(self, name: str, dimension: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Implementation-specific collection creation."""
        raise NotImplementedError
    
    async def _delete_collection_impl(self, name: str) -> bool:
        """Implementation-specific collection deletion."""
        raise NotImplementedError
    
    async def _list_collections_impl(self) -> List[str]:
        """Implementation-specific collection listing."""
        raise NotImplementedError
    
    async def _collection_exists_impl(self, name: str) -> bool:
        """Implementation-specific collection existence check."""
        raise NotImplementedError
    
    async def _store_documents_impl(self, documents: List[Document], collection: str) -> List[str]:
        """Implementation-specific document storage."""
        raise NotImplementedError
    
    async def _similarity_search_impl(
        self,
        query_embedding: List[float],
        collection: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Implementation-specific similarity search."""
        raise NotImplementedError
    
    async def _get_document_impl(self, doc_id: str, collection: str) -> Optional[Document]:
        """Implementation-specific document retrieval."""
        raise NotImplementedError
    
    async def _delete_documents_impl(self, doc_ids: List[str], collection: str) -> bool:
        """Implementation-specific document deletion."""
        raise NotImplementedError
    
    async def _update_document_impl(self, doc_id: str, document: Document, collection: str) -> bool:
        """Implementation-specific document update."""
        raise NotImplementedError
    
    async def _count_documents_impl(self, collection: str) -> int:
        """Implementation-specific document counting."""
        raise NotImplementedError