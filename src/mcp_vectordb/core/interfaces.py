"""Abstract base classes and interfaces for MCP Vector DB Server."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from .document import Document, SearchResult


class EmbeddingService(ABC):
    """Abstract base class for embedding services."""
    
    @abstractmethod
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text.
        
        Args:
            text: The text to embed
            
        Returns:
            The embedding vector
        """
        pass
    
    @abstractmethod
    async def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        pass
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Get the dimension of the embedding vectors."""
        pass
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Get the name of the embedding model."""
        pass


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


class DocumentProcessor(ABC):
    """Abstract base class for document processors."""
    
    @abstractmethod
    async def process_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> Document:
        """Process text into a document.
        
        Args:
            text: The text to process
            metadata: Optional metadata
            
        Returns:
            Processed document
        """
        pass
    
    @abstractmethod
    async def process_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> List[Document]:
        """Process multiple texts into documents.
        
        Args:
            texts: List of texts to process
            metadatas: Optional list of metadata dictionaries
            
        Returns:
            List of processed documents
        """
        pass