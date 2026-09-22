from fastapi import APIRouter, HTTPException
import uuid
from ..schemas.documents import DocumentCreate, DocumentOut
from pathlib import Path
from vectorforge.core.ingestion.pipeline import ingest_document
from vectorforge.core.embeddings.provider import EmbeddingProvider

router = APIRouter()

# in-memory document store
_DOCUMENTS = {}


@router.post("/collections/{collection_id}/documents", response_model=DocumentOut)
def add_document(collection_id: str, req: DocumentCreate):
    did = str(uuid.uuid4())
    _DOCUMENTS[did] = {"id": did, "external_id": req.external_id, "title": req.title, "metadata": req.metadata}
    # Note: embedding and chunking are not persisted here; use CLI ingest for now
    return _DOCUMENTS[did]


@router.get("/documents/{document_id}")
def get_document(document_id: str):
    doc = _DOCUMENTS.get(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc
