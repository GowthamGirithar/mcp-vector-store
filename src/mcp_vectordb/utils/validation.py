"""Input validation utilities for MCP Vector DB Server."""

import os
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
    reserved_keys = {
        "id", "embedding", "document", "text",
        # Reserved for document-upload-tool auto-generated metadata fields.
        "document_id", "source_filename", "file_type",
        "page_number", "chunk_index", "total_chunks", "uploaded_at",
    }
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


def validate_file_path(path: str, allowed_extensions: set, max_file_size_mb: float) -> str:
    """Validate a file path for document upload.

    Args:
        path: The file path to validate
        allowed_extensions: Set of allowed file extensions (e.g. {".txt", ".pdf"})
        max_file_size_mb: Maximum allowed file size in megabytes

    Returns:
        The validated file path

    Raises:
        ValidationError: If the file path is invalid
    """
    if not isinstance(path, str):
        raise ValidationError("File path must be a string")

    if not path.strip():
        raise ValidationError("File path cannot be empty")

    stripped_path = path.strip()

    if not os.path.isfile(stripped_path):
        raise ValidationError(f"File not found: {stripped_path}")

    if not os.access(stripped_path, os.R_OK):
        raise ValidationError(f"File is not readable: {stripped_path}")

    _, extension = os.path.splitext(stripped_path)
    extension = extension.lower()
    if extension not in allowed_extensions:
        allowed_str = ", ".join(sorted(allowed_extensions))
        raise ValidationError(
            f"Unsupported file extension '{extension}'. Allowed: {allowed_str}"
        )

    file_size_mb = os.path.getsize(stripped_path) / (1024 * 1024)
    if file_size_mb > max_file_size_mb:
        raise ValidationError(
            f"File size ({file_size_mb:.1f} MB) exceeds maximum ({max_file_size_mb:.1f} MB)"
        )

    return stripped_path


def validate_chunk_size(chunk_size: int) -> int:
    """Validate the chunk size used for document splitting.

    Args:
        chunk_size: The chunk size to validate

    Returns:
        The validated chunk size

    Raises:
        ValidationError: If chunk size is invalid
    """
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise ValidationError("Chunk size must be an integer")

    if chunk_size <= 0:
        raise ValidationError("Chunk size must be positive")

    return chunk_size


def validate_chunk_overlap(chunk_overlap: int, chunk_size: int) -> int:
    """Validate the chunk overlap used for document splitting.

    Args:
        chunk_overlap: The chunk overlap to validate
        chunk_size: The chunk size the overlap must be smaller than

    Returns:
        The validated chunk overlap

    Raises:
        ValidationError: If chunk overlap is invalid
    """
    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise ValidationError("Chunk overlap must be an integer")

    if chunk_overlap < 0:
        raise ValidationError("Chunk overlap cannot be negative")

    if chunk_overlap >= chunk_size:
        raise ValidationError(
            f"Chunk overlap ({chunk_overlap}) must be less than chunk size ({chunk_size})"
        )

    return chunk_overlap


_ALLOWED_CHUNK_STRATEGIES = {"recursive", "structural"}


def validate_chunk_strategy(chunk_strategy: str) -> str:
    """Validate the chunk strategy used for document splitting.

    Args:
        chunk_strategy: The chunk strategy to validate

    Returns:
        The validated chunk strategy

    Raises:
        ValidationError: If chunk strategy is not one of the allowed values
    """
    if chunk_strategy not in _ALLOWED_CHUNK_STRATEGIES:
        allowed_str = ", ".join(sorted(_ALLOWED_CHUNK_STRATEGIES))
        raise ValidationError(
            f"Invalid chunk strategy '{chunk_strategy}'. Allowed: {allowed_str}"
        )

    return chunk_strategy


def _is_json_serializable(value: Any) -> bool:
    """Check if a value is JSON serializable."""
    import json
    try:
        json.dumps(value)
        return True
    except (TypeError, ValueError):
        return False