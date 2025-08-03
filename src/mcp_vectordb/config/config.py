"""Simple configuration settings using environment variables."""

import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

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


class Settings(BaseModel):
    """Main settings class loaded from environment variables."""
    vector_db: VectorDBConfig
    embedding: EmbeddingConfig

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
        
        return cls(vector_db=vector_db, embedding=embedding)


# Global settings instance
# Optional[Settings] == Union[Settings, None] - so optional is required to assign None
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the global settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings