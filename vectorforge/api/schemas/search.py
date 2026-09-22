from pydantic import BaseModel
from typing import Optional, Dict, Any


class SearchRequest(BaseModel):
    collection_id: str
    query: str
    top_k: int = 10
    method: str = "hnsw"
    parameters: Optional[Dict[str, Any]] = None


class SearchResultItem(BaseModel):
    chunk_id: str
    document_id: Optional[str]
    score: float
    text: Optional[str]


class SearchResponse(BaseModel):
    query: str
    method: str
    latency_ms: float
    results: list[SearchResultItem]
    metadata: Optional[Dict[str, Any]] = None
