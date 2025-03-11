from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class IndexType(str, Enum):
    FLAT = "flat"
    IVF = "ivf"
    HNSW = "hnsw"
    PQ = "pq"
    IVF_PQ = "ivf_pq"
    SQ = "sq"


class SearchParams(BaseModel):
    """Parameters specific to different index types"""

    ef_search: Optional[int] = Field(50, description="HNSW efSearch parameter")
    nprobe: Optional[int] = Field(10, description="IVF nprobe parameter")


class SearchRequest(BaseModel):
    query: str
    k: int = 5
    index_type: Optional[IndexType] = None
    params: Optional[SearchParams] = None


class SearchResult(BaseModel):
    document_id: str
    content: str
    distance: float
    metadata: Optional[Dict[str, Any]] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    index_type: IndexType
    search_time_ms: float
    total_results: int
