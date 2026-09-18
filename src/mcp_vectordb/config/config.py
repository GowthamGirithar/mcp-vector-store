"""Simple configuration settings using environment variables."""

import os
from typing import Optional
from pydantic import BaseModel, Field, model_validator
from dotenv import load_dotenv

from ..chunking.process_document import SUPPORTED_PARSERS
from ..utils.validation import validate_chunk_size, validate_chunk_overlap
from ..utils.exceptions import ValidationError

# Load environment variables from .env file
load_dotenv()


class VectorDBConfig(BaseModel):
    """Vector database configuration."""
    provider: str = Field(default="chroma")
    host: str = Field(default="localhost")
    port: int = Field(default=8000)
    path: str = Field(default="./chroma_db")


class EmbeddingConfig(BaseModel):
    """Embedding service configuration."""
    provider: str = Field(default="sentence-transformers")
    model: str = Field(default="all-MiniLM-L6-v2")
    api_key: Optional[str] = Field(default=None)


class LLMConfig(BaseModel):
    """LLM completion service configuration, used for agentic embedding."""
    provider: str = Field(default="openai")
    model: str = Field(default="gpt-4o-mini")
    api_key: Optional[str] = Field(default=None)
    # Rough proxy for staying under the model's context window (default sized
    # for gpt-4o-mini's ~128k token context, leaving headroom for the LLM's
    # output since it may echo chunk text back). Documents over this size
    # fall back to per-title-section calls; see `core/agentic_chunking.py`.
    max_input_chars: int = Field(default=300000)


class ServerConfig(BaseModel):
    """Server configuration for MCP server."""
    transport: str = Field(default="streamable-http")


class DocumentConfig(BaseModel):
    """Document upload/chunking configuration."""
    chunk_size: int = Field(default=500)
    chunk_overlap: int = Field(default=50)
    max_file_size_mb: float = Field(default=20.0)
    parser: str = Field(default="unstructured")

    @model_validator(mode="after")
    def _validate_chunking(self) -> "DocumentConfig":
        try:
            validate_chunk_size(self.chunk_size)
            validate_chunk_overlap(self.chunk_overlap, self.chunk_size)
        except ValidationError as e:
            raise ValueError(e.message)
        if self.parser not in SUPPORTED_PARSERS:
            raise ValueError(
                f"Unknown parser '{self.parser}'. Supported parsers are: {', '.join(SUPPORTED_PARSERS)}"
            )
        return self


class CompressionConfig(BaseModel):
    """TurboQuant-style embedding compression configuration.

    Off by default: enabling it changes what gets stored per vector, so
    turning it on for an existing collection requires re-indexing (encoding
    codes for already-stored embeddings) rather than just a config flip.
    shadow_enabled runs the compressed retrieval path alongside the real
    search path for comparison, without it affecting returned results.
    """
    enabled: bool = Field(default=False)
    shadow_enabled: bool = Field(default=False)
    bits: int = Field(default=8)
    seed: int = Field(default=42)
    candidate_pool_size: int = Field(default=200)


class SearchConfig(BaseModel):
    """Search/retrieval tuning configuration.

    default_top_k and default_min_score are per-query fallbacks: tools still
    accept them as optional call-time overrides. The RRF/reranker knobs are
    deployment-level tuning and are intentionally not exposed as tool inputs.
    """
    default_top_k: int = Field(default=10)
    default_min_score: Optional[float] = Field(default=None)
    rrf_k: int = Field(default=60)
    vector_weight: float = Field(default=1.0)
    bm25_weight: float = Field(default=1.0)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    use_reranker: bool = Field(default=False)


class Settings(BaseModel):
    """Main settings class loaded from environment variables."""
    vector_db: VectorDBConfig
    embedding: EmbeddingConfig
    llm: LLMConfig
    server: ServerConfig
    document: DocumentConfig
    search: SearchConfig
    compression: CompressionConfig

    @classmethod
    def from_env(cls) -> "Settings":
        """Load settings from environment variables."""
        vector_db = VectorDBConfig(
            provider=os.getenv("VECTOR_DB_PROVIDER", "chroma"),
            host=os.getenv("VECTOR_DB_HOST", "localhost"),
            port=int(os.getenv("VECTOR_DB_PORT", "8000")),
            path=os.getenv("VECTOR_DB_PATH", "./chroma_db")
        )
        
        embedding = EmbeddingConfig(
            provider=os.getenv("EMBEDDING_PROVIDER", "openai"),
            model=os.getenv("EMBEDDING_MODEL", "text-embedding-ada-002"),
            api_key=os.getenv("OPENAI_API_KEY")
        )
        
        llm = LLMConfig(
            provider=os.getenv("LLM_PROVIDER", "openai"),
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            max_input_chars=int(os.getenv("LLM_MAX_INPUT_CHARS", "300000")),
        )

        server = ServerConfig(
            transport=os.getenv("MCP_TRANSPORT", "streamable-http")
        )

        document = DocumentConfig(
            chunk_size=int(os.getenv("DOCUMENT_CHUNK_SIZE", "500")),
            chunk_overlap=int(os.getenv("DOCUMENT_CHUNK_OVERLAP", "50")),
            max_file_size_mb=float(os.getenv("DOCUMENT_MAX_FILE_SIZE_MB", "20.0")),
            parser=os.getenv("DOCUMENT_PARSER", "unstructured"),
        )

        default_min_score_env = os.getenv("SEARCH_DEFAULT_MIN_SCORE")
        search = SearchConfig(
            default_top_k=int(os.getenv("SEARCH_DEFAULT_TOP_K", "10")),
            default_min_score=float(default_min_score_env) if default_min_score_env is not None else None,
            rrf_k=int(os.getenv("SEARCH_RRF_K", "60")),
            vector_weight=float(os.getenv("SEARCH_VECTOR_WEIGHT", "1.0")),
            bm25_weight=float(os.getenv("SEARCH_BM25_WEIGHT", "1.0")),
            reranker_model=os.getenv("SEARCH_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
            use_reranker=os.getenv("SEARCH_USE_RERANKER", "false").lower() in ("1", "true", "yes")
        )

        compression = CompressionConfig(
            enabled=os.getenv("COMPRESSION_ENABLED", "false").lower() in ("1", "true", "yes"),
            shadow_enabled=os.getenv("COMPRESSION_SHADOW_ENABLED", "false").lower() in ("1", "true", "yes"),
            bits=int(os.getenv("COMPRESSION_BITS", "8")),
            seed=int(os.getenv("COMPRESSION_SEED", "42")),
            candidate_pool_size=int(os.getenv("COMPRESSION_CANDIDATE_POOL_SIZE", "200")),
        )

        return cls(
            vector_db=vector_db, embedding=embedding, llm=llm, server=server,
            document=document, search=search, compression=compression,
        )


# Global settings instance
# Optional[Settings] == Union[Settings, None] - so optional is required to assign None
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings