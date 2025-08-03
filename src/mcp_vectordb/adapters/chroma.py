"""ChromaDB adapter implementation for MCP Vector DB Server."""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from chromadb.api.models.Collection import Collection

from .base import BaseVectorDBAdapter
from ..core.document import Document, SearchResult
from ..utils.exceptions import VectorDBError, ConnectionError, CollectionError
from ..config.config import VectorDBConfig

logger = logging.getLogger(__name__)


class ChromaAdapter(BaseVectorDBAdapter):
    """ChromaDB adapter implementation."""
    
    def __init__(self, config: VectorDBConfig):
        """Initialize Chroma adapter.
        
        Args:
            config: Vector database configuration
        """
        super().__init__(config.dict())
        self.config = config
        self.client = None
        self._collections_cache = {}
    
    async def _connect(self) -> None:
        """Connect to ChromaDB."""
        try:
            # Determine connection type based on configuration
            if self.config.host == "localhost" and hasattr(self.config, 'path'):
                # Local persistent client
                chroma_settings = ChromaSettings(
                    persist_directory=self.config.path,
                    anonymized_telemetry=False
                )
                self.client = chromadb.PersistentClient(
                    path=self.config.path,
                    settings=chroma_settings
                )
                logger.info(f"Connected to local ChromaDB at {self.config.path}")
            else:
                # HTTP client
                self.client = chromadb.HttpClient(
                    host=self.config.host,
                    port=self.config.port
                )
                logger.info(f"Connected to ChromaDB server at {self.config.host}:{self.config.port}")
            
            # Test connection
            await self._test_connection()
            
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise ConnectionError(f"ChromaDB connection failed: {e}")
    
    async def _disconnect(self) -> None:
        """Disconnect from ChromaDB."""
        if self.client:
            # ChromaDB doesn't require explicit disconnection
            self.client = None
            self._collections_cache.clear()
            logger.info("Disconnected from ChromaDB")
    
    async def _test_connection(self) -> None:
        """Test ChromaDB connection."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self.client.heartbeat)
        except Exception as e:
            raise ConnectionError(f"ChromaDB connection test failed: {e}")
    
    async def _health_check_impl(self) -> bool:
        """Implementation-specific health check."""
        try:
            await self._test_connection()
            return True
        except Exception:
            return False
    
    async def _create_collection_impl(self, name: str, dimension: int, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Create a new collection in ChromaDB."""
        try:
            loop = asyncio.get_event_loop()
            
            # Prepare collection metadata
            collection_metadata = {
                "dimension": dimension,
                "created_at": datetime.utcnow().isoformat()
            }
            if metadata:
                collection_metadata.update(metadata)
            
            # Create collection
            collection = await loop.run_in_executor(
                None,
                lambda: self.client.create_collection(
                    name=name,
                    metadata=collection_metadata
                )
            )
            
            # Cache the collection
            self._collections_cache[name] = collection
            
            logger.info(f"Created ChromaDB collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create ChromaDB collection {name}: {e}")
            raise CollectionError(f"Collection creation failed: {e}")
    
    async def _delete_collection_impl(self, name: str) -> bool:
        """Delete a collection from ChromaDB."""
        try:
            loop = asyncio.get_event_loop()
            
            await loop.run_in_executor(
                None,
                lambda: self.client.delete_collection(name=name)
            )
            
            # Remove from cache
            self._collections_cache.pop(name, None)
            
            logger.info(f"Deleted ChromaDB collection: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete ChromaDB collection {name}: {e}")
            raise CollectionError(f"Collection deletion failed: {e}")
    
    async def _list_collections_impl(self) -> List[str]:
        """List all collections in ChromaDB."""
        try:
            loop = asyncio.get_event_loop()
            
            collections = await loop.run_in_executor(
                None,
                self.client.list_collections
            )
            
            collection_names = [col.name for col in collections]
            logger.debug(f"Listed {len(collection_names)} ChromaDB collections")
            return collection_names
            
        except Exception as e:
            logger.error(f"Failed to list ChromaDB collections: {e}")
            raise CollectionError(f"Collection listing failed: {e}")
    
    async def _collection_exists_impl(self, name: str) -> bool:
        """Check if a collection exists in ChromaDB."""
        try:
            collections = await self._list_collections_impl()
            return name in collections
        except Exception as e:
            logger.error(f"Failed to check ChromaDB collection existence {name}: {e}")
            return False
    
    async def _get_collection(self, name: str) -> Collection:
        """Get a ChromaDB collection object."""
        if name in self._collections_cache:
            return self._collections_cache[name]
        
        try:
            loop = asyncio.get_event_loop()
            collection = await loop.run_in_executor(
                None,
                lambda: self.client.get_collection(name=name)
            )
            
            # Cache the collection
            self._collections_cache[name] = collection
            return collection
            
        except Exception as e:
            logger.error(f"Failed to get ChromaDB collection {name}: {e}")
            raise CollectionError(f"Collection not found: {name}")
    
    async def _store_documents_impl(self, documents: List[Document], collection: str) -> List[str]:
        """Store documents in ChromaDB."""
        try:
            chroma_collection = await self._get_collection(collection)
            
            # Prepare data for ChromaDB
            ids = [doc.id for doc in documents]
            embeddings = [doc.embedding for doc in documents]
            documents_text = [doc.text for doc in documents]
            metadatas = []
            
            for doc in documents:
                metadata = doc.metadata.copy() if doc.metadata else {}
                metadata.update({
                    "created_at": doc.created_at.isoformat(),
                })
                metadatas.append(metadata)
            
            # Store in ChromaDB
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: chroma_collection.add(
                    ids=ids,
                    embeddings=embeddings,
                    documents=documents_text,
                    metadatas=metadatas
                )
            )
            
            logger.info(f"Stored {len(documents)} documents in ChromaDB collection {collection}")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to store documents in ChromaDB collection {collection}: {e}")
            raise VectorDBError(f"Document storage failed: {e}")
    
    async def _similarity_search_impl(
        self,
        query_embedding: List[float],
        collection: str,
        top_k: int,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[SearchResult]:
        """Perform similarity search in ChromaDB."""
        try:
            chroma_collection = await self._get_collection(collection)
            
            # Prepare query parameters
            query_params = {
                "query_embeddings": [query_embedding],
                "n_results": top_k
            }
            
            # Add filters if provided
            if filters:
                query_params["where"] = filters
            
            # Perform search
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: chroma_collection.query(**query_params)
            )
            
            # Convert results to SearchResult objects
            search_results = []
            
            if results["ids"] and results["ids"][0]:  # Check if we have results
                for i in range(len(results["ids"][0])):
                    doc_id = results["ids"][0][i]
                    distance = results["distances"][0][i] if results["distances"] else None
                    document_text = results["documents"][0][i] if results["documents"] else ""
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    
                    # Convert distance to similarity score (ChromaDB uses cosine distance)
                    score = 1.0 - distance if distance is not None else 1.0
                    
                    # Create Document object
                    doc_metadata = metadata.copy()
                    created_at = doc_metadata.pop("created_at", datetime.utcnow().isoformat())
                    updated_at = doc_metadata.pop("updated_at", None)
                    
                    document = Document(
                        id=doc_id,
                        text=document_text,
                        metadata=doc_metadata,
                        created_at=datetime.fromisoformat(created_at),
                        updated_at=datetime.fromisoformat(updated_at) if updated_at else None
                    )
                    
                    search_result = SearchResult(
                        document=document,
                        score=score,
                        distance=distance
                    )
                    
                    search_results.append(search_result)
            
            logger.debug(f"Found {len(search_results)} results in ChromaDB collection {collection}")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search ChromaDB collection {collection}: {e}")
            raise VectorDBError(f"Similarity search failed: {e}")
    
    async def _get_document_impl(self, doc_id: str, collection: str) -> Optional[Document]:
        """Get a document by ID from ChromaDB."""
        try:
            chroma_collection = await self._get_collection(collection)
            
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: chroma_collection.get(ids=[doc_id], include=["documents", "metadatas", "embeddings"])
            )
            
            if not results["ids"] or not results["ids"]:
                return None
            
            # Extract document data
            document_text = results["documents"][0] if results["documents"] else ""
            metadata = results["metadatas"][0] if results["metadatas"] else {}
            embedding = results["embeddings"][0] if results["embeddings"] else None
            
            # Create Document object
            doc_metadata = metadata.copy()
            created_at = doc_metadata.pop("created_at", datetime.utcnow().isoformat())
            updated_at = doc_metadata.pop("updated_at", None)
            
            document = Document(
                id=doc_id,
                text=document_text,
                embedding=embedding,
                metadata=doc_metadata,
                created_at=datetime.fromisoformat(created_at),
                updated_at=datetime.fromisoformat(updated_at) if updated_at else None
            )
            
            logger.debug(f"Retrieved document {doc_id} from ChromaDB collection {collection}")
            return document
            
        except Exception as e:
            logger.error(f"Failed to get document {doc_id} from ChromaDB collection {collection}: {e}")
            raise VectorDBError(f"Document retrieval failed: {e}")
    
    async def _delete_documents_impl(self, doc_ids: List[str], collection: str) -> bool:
        """Delete documents by IDs from ChromaDB."""
        try:
            chroma_collection = await self._get_collection(collection)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: chroma_collection.delete(ids=doc_ids)
            )
            
            logger.info(f"Deleted {len(doc_ids)} documents from ChromaDB collection {collection}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete documents from ChromaDB collection {collection}: {e}")
            raise VectorDBError(f"Document deletion failed: {e}")
    
    async def _update_document_impl(self, doc_id: str, document: Document, collection: str) -> bool:
        """Update a document in ChromaDB."""
        try:
            chroma_collection = await self._get_collection(collection)
            
            # Prepare metadata
            metadata = document.metadata.copy() if document.metadata else {}
            metadata.update({
                "created_at": document.created_at.isoformat(),
                "updated_at": datetime.utcnow().isoformat()
            })
            
            # Update in ChromaDB
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: chroma_collection.update(
                    ids=[doc_id],
                    embeddings=[document.embedding] if document.embedding else None,
                    documents=[document.text],
                    metadatas=[metadata]
                )
            )
            
            logger.info(f"Updated document {doc_id} in ChromaDB collection {collection}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update document {doc_id} in ChromaDB collection {collection}: {e}")
            raise VectorDBError(f"Document update failed: {e}")
    
    async def _count_documents_impl(self, collection: str) -> int:
        """Count documents in a ChromaDB collection."""
        try:
            chroma_collection = await self._get_collection(collection)
            
            loop = asyncio.get_event_loop()
            count = await loop.run_in_executor(
                None,
                lambda: chroma_collection.count()
            )
            
            logger.debug(f"ChromaDB collection {collection} has {count} documents")
            return count
            
        except Exception as e:
            logger.error(f"Failed to count documents in ChromaDB collection {collection}: {e}")
            raise VectorDBError(f"Document count failed: {e}")