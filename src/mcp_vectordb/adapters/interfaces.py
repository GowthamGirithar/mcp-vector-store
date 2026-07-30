"""Abstract interface for vector database adapters."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from ..models.document import Document, SearchResult


class VectorDBAdapter(ABC):
    """Abstract base class for vector database adapters."""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the vector database connection."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close the vector database connection."""
        pass

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the vector database is healthy."""
        pass

    @abstractmethod
    async def create_collection(self, name: str, dimension: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection.

        Args:
            name: Collection name
            dimension: Vector dimension
            metadata: Optional collection metadata

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def delete_collection(self, name: str) -> bool:
        """Delete a collection.

        Args:
            name: Collection name

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def list_collections(self) -> List[str]:
        """List all collections.

        Returns:
            List of collection names
        """
        pass

    @abstractmethod
    async def collection_exists(self, name: str) -> bool:
        """Check if a collection exists.

        Args:
            name: Collection name

        Returns:
            True if collection exists
        """
        pass

    @abstractmethod
    async def store_documents(self, documents: List[Document], collection: str) -> List[str]:
        """Store documents in the vector database.

        Args:
            documents: List of documents to store
            collection: Collection name

        Returns:
            List of document IDs
        """
        pass

    @abstractmethod
    async def similarity_search(
        self,
        query_embedding: List[float],
        collection: str,
        top_k: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search.

        Args:
            query_embedding: Query vector
            collection: Collection name
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results
        """
        pass

    @abstractmethod
    async def get_document(self, doc_id: str, collection: str) -> Optional[Document]:
        """Get a document by ID.

        Args:
            doc_id: Document ID
            collection: Collection name

        Returns:
            Document if found, None otherwise
        """
        pass

    @abstractmethod
    async def delete_documents(self, doc_ids: List[str], collection: str) -> bool:
        """Delete documents by IDs.

        Args:
            doc_ids: List of document IDs
            collection: Collection name

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def update_document(self, doc_id: str, document: Document, collection: str) -> bool:
        """Update a document.

        Args:
            doc_id: Document ID
            document: Updated document
            collection: Collection name

        Returns:
            True if successful
        """
        pass

    @abstractmethod
    async def count_documents(self, collection: str) -> int:
        """Count documents in a collection.

        Args:
            collection: Collection name

        Returns:
            Number of documents
        """
        pass
