from pydantic import BaseModel
from typing import Optional


class CollectionCreate(BaseModel):
    name: str
    description: Optional[str] = None
    embedding_model: Optional[str] = None
    embedding_dimension: int
    distance_metric: Optional[str] = "cosine"


class CollectionOut(BaseModel):
    id: str
    name: str
    description: Optional[str]
    embedding_model: Optional[str]
    embedding_dimension: int
    distance_metric: str
