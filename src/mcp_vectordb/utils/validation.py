"""Input validation utilities for MCP Vector DB Server."""

import re
from typing import Dict, Any, Optional
from .exceptions import ValidationError


def validate_text(text: str, max_length: int = 100000) -> str:
    """Validate text input for storage.
    
    Args:
        text: The text to validate
        max_length: Maximum allowed text length
        
    Returns:
        The validated text
        
    Raises:
        ValidationError: If text is invalid
    """
    if not isinstance(text, str):
        raise ValidationError("Text must be a string")
    
    if not text.strip():
        raise ValidationError("Text cannot be empty or only whitespace")
    
    if len(text) > max_length:
        raise ValidationError(f"Text length ({len(text)}) exceeds maximum ({max_length})")
    
    return text.strip()


def validate_metadata(metadata: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate metadata dictionary.
    
    Args:
        metadata: The metadata to validate
        
    Returns:
        The validated metadata
        
    Raises:
        ValidationError: If metadata is invalid
    """
    if metadata is None:
        return None
    
    if not isinstance(metadata, dict):
        raise ValidationError("Metadata must be a dictionary")
    
    # Check for reserved keys
    reserved_keys = {"id", "embedding", "document", "text"}
    for key in metadata.keys():
        if key in reserved_keys:
            raise ValidationError(f"Metadata key '{key}' is reserved")
    
    # Validate metadata values
    for key, value in metadata.items():
        if not isinstance(key, str):
            raise ValidationError("Metadata keys must be strings")
        
        # Check if value is JSON serializable
        if not _is_json_serializable(value):
            raise ValidationError(f"Metadata value for key '{key}' is not JSON serializable")
        
    print(f"Validated metadata: {metadata}")
    
    return metadata


def validate_collection_name(collection_name: str) -> str:
    """Validate collection name.
    
    Args:
        collection_name: The collection name to validate
        
    Returns:
        The validated collection name
        
    Raises:
        ValidationError: If collection name is invalid
    """
    if not isinstance(collection_name, str):
        raise ValidationError("Collection name must be a string")
    
    if not collection_name.strip():
        raise ValidationError("Collection name cannot be empty")
    
    # Check length
    if len(collection_name) > 63:
        raise ValidationError("Collection name cannot exceed 63 characters")
    
    # Check format (alphanumeric, hyphens, underscores)
    if not re.match(r'^[a-zA-Z0-9_-]+$', collection_name):
        raise ValidationError(
            "Collection name can only contain letters, numbers, hyphens, and underscores"
        )
    
    # Cannot start or end with hyphen or underscore
    if collection_name.startswith(('-', '_')) or collection_name.endswith(('-', '_')):
        raise ValidationError(
            "Collection name cannot start or end with hyphen or underscore"
        )
    
    return collection_name.strip()


def validate_top_k(top_k: int, max_k: int = 1000) -> int:
    """Validate top_k parameter for similarity search.
    
    Args:
        top_k: Number of results to return
        max_k: Maximum allowed value
        
    Returns:
        The validated top_k value
        
    Raises:
        ValidationError: If top_k is invalid
    """
    if not isinstance(top_k, int):
        raise ValidationError("top_k must be an integer")
    
    if top_k <= 0:
        raise ValidationError("top_k must be positive")
    
    if top_k > max_k:
        raise ValidationError(f"top_k ({top_k}) exceeds maximum ({max_k})")
    
    return top_k


def validate_document_id(doc_id: str) -> str:
    """Validate document ID.
    
    Args:
        doc_id: The document ID to validate
        
    Returns:
        The validated document ID
        
    Raises:
        ValidationError: If document ID is invalid
    """
    if not isinstance(doc_id, str):
        raise ValidationError("Document ID must be a string")
    
    if not doc_id.strip():
        raise ValidationError("Document ID cannot be empty")
    
    if len(doc_id) > 255:
        raise ValidationError("Document ID cannot exceed 255 characters")
    
    return doc_id.strip()


def _is_json_serializable(value: Any) -> bool:
    """Check if a value is JSON serializable."""
    import json
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False