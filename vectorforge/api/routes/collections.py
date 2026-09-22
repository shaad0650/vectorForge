from fastapi import APIRouter, HTTPException, Depends
import uuid
from ..schemas.collections import CollectionCreate, CollectionOut
from pathlib import Path
import json
try:
    from vectorforge.db import SessionLocal, Collection as DBCollection
    DB_AVAILABLE = True
except Exception:
    SessionLocal = None
    DBCollection = None
    DB_AVAILABLE = False

router = APIRouter()


if DB_AVAILABLE:
    def get_db():
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
else:
    def get_db():
        # simple stub when DB not available
        yield None


@router.post("/collections", response_model=CollectionOut)
def create_collection(req: CollectionCreate, db: SessionLocal = Depends(get_db)):
    cid = str(uuid.uuid4())
    item = {"id": cid, "name": req.name, "description": req.description, "embedding_model": req.embedding_model, "embedding_dimension": req.embedding_dimension, "distance_metric": req.distance_metric}
    if DB_AVAILABLE and db is not None:
        dbc = DBCollection(id=cid, name=req.name, description=req.description, embedding_model=req.embedding_model, embedding_dimension=req.embedding_dimension, distance_metric=req.distance_metric)
        db.add(dbc)
        db.commit()
    else:
        # fallback to in-memory registry for tests
        _COLLECTIONS[cid] = item
    return item


@router.get("/collections")
def list_collections(db: SessionLocal = Depends(get_db)):
    if DB_AVAILABLE and db is not None:
        rows = db.query(DBCollection).all()
        out = []
        for r in rows:
            out.append({"id": r.id, "name": r.name, "description": r.description, "embedding_model": r.embedding_model, "embedding_dimension": r.embedding_dimension, "distance_metric": r.distance_metric})
        return out
    else:
        return list(_COLLECTIONS.values())


@router.get("/collections/{collection_id}")
def get_collection(collection_id: str, db: SessionLocal = Depends(get_db)):
    if DB_AVAILABLE and db is not None:
        item = db.query(DBCollection).filter(DBCollection.id == collection_id).first()
        if not item:
            raise HTTPException(status_code=404, detail="Collection not found")
        return {"id": item.id, "name": item.name, "description": item.description, "embedding_model": item.embedding_model, "embedding_dimension": item.embedding_dimension, "distance_metric": item.distance_metric}
    else:
        item = _COLLECTIONS.get(collection_id)
        if not item:
            raise HTTPException(status_code=404, detail="Collection not found")
        return item



@router.post("/collections/{collection_id}/load")
def load_collection(collection_id: str, db: SessionLocal = Depends(get_db)):
    # check DB
    item = db.query(DBCollection).filter(DBCollection.id == collection_id).first()
    if not item:
        # try disk registry
        base = Path("data") / "collections"
        registry_path = base / "collections.json"
        if registry_path.exists():
            try:
                with open(registry_path, "r", encoding="utf-8") as f:
                    reg = json.load(f)
                entry = reg.get(collection_id)
                if entry:
                    # persist minimal info
                    dbc = DBCollection(id=entry.get("id", collection_id), name=entry.get("name", collection_id), description=entry.get("description"))
                    db.add(dbc)
                    db.commit()
                else:
                    raise HTTPException(status_code=404, detail="Collection not found")
            except HTTPException:
                raise
            except Exception:
                raise HTTPException(status_code=500, detail="Failed to read registry")
        else:
            raise HTTPException(status_code=404, detail="Collection not found")
    # attempt to load npz into search module via import
    from . import search as search_mod
    ok = search_mod._load_collection_npz_if_present(collection_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Collection .npz not found on disk")
    _LOADED.add(collection_id)
    return {"collection_id": collection_id, "status": "loaded"}


@router.post("/collections/{collection_id}/unload")
def unload_collection(collection_id: str):
    if collection_id in _LOADED:
        _LOADED.remove(collection_id)
    # also remove from in-memory embeddings
    from . import search as search_mod
    if collection_id in search_mod._EMBEDDINGS:
        del search_mod._EMBEDDINGS[collection_id]
    return {"collection_id": collection_id, "status": "unloaded"}


@router.get("/collections/registry")
def list_collections_registry():
    base = Path("data") / "collections"
    registry_path = base / "collections.json"
    if not registry_path.exists():
        return {}
    try:
        with open(registry_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception:
        # on error, return empty
        return {}
