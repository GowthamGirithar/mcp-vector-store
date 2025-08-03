"""Document and search result data models."""

import uuid
from datetime import datetime
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class Document(BaseModel):
    """Document model for storing text with embeddings and metadata."""
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: str = Field(..., description="The document text content")
    embedding: Optional[List[float]] = Field(default=None, description="The embedding vector")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Document metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = Field(default=None)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert document to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "embedding": self.embedding,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }
    


class SearchResult(BaseModel):
    """Search result model containing document and similarity score."""
    
    document: Document = Field(..., description="The matched document")
    score: float = Field(..., description="Similarity score (0.0 to 1.0)")
    distance: Optional[float] = Field(default=None, description="Distance metric (optional)")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert search result to dictionary."""
        return {
            "document": self.document.to_dict(),
            "score": self.score,
            "distance": self.distance
        }
    


class CollectionInfo(BaseModel):
    """Collection information model."""
    
    name: str = Field(..., description="Collection name")
    dimension: int = Field(..., description="Vector dimension")
    document_count: int = Field(default=0, description="Number of documents")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Collection metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert collection info to dictionary."""
        return {
            "name": self.name,
            "dimension": self.dimension,
            "document_count": self.document_count,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat()
        }


class QueryRequest(BaseModel):
    """Query request model for similarity search."""
    
    query: str = Field(..., description="Query text")
    collection: str = Field(..., description="Collection to search")
    top_k: int = Field(default=10, description="Number of results to return")
    filters: Optional[Dict[str, Any]] = Field(default=None, description="Metadata filters")
    include_embeddings: bool = Field(default=False, description="Include embeddings in results")
    
    def validate_top_k(self, max_k: int = 1000) -> None:
        """Validate top_k parameter."""
        if self.top_k <= 0:
            raise ValueError("top_k must be positive")
        if self.top_k > max_k:
            raise ValueError(f"top_k cannot exceed {max_k}")