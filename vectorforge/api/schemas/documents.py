from pydantic import BaseModel
from typing import Optional, Dict, Any


class DocumentCreate(BaseModel):
    external_id: Optional[str]
    title: Optional[str]
    text: str
    metadata: Optional[Dict[str, Any]] = None


class DocumentOut(BaseModel):
    id: str
    external_id: Optional[str]
    title: Optional[str]
    metadata: Optional[Dict[str, Any]]
