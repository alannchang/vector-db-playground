from typing import List
from pydantic_settings import BaseSettings
from app.models.search import IndexType


class Settings(BaseSettings):
    APP_NAME: str = "Vector Search API"
    DEBUG: bool = False

    # Embedding model settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Index settings
    DEFAULT_INDEX_TYPE: IndexType = IndexType.FLAT
    ENABLED_INDICES: List[IndexType] = [
        IndexType.FLAT,
        IndexType.HNSW,
        IndexType.IVF,
        # IndexType.PQ,      # Temporarily disabled
        # IndexType.IVF_PQ,  # Temporarily disabled
        IndexType.SQ,
    ]

    # Default search parameters
    DEFAULT_HNSW_EF_SEARCH: int = 50
    DEFAULT_IVF_NPROBE: int = 10

    class Config:
        env_file = ".env"


settings = Settings()
