from fastapi import APIRouter, HTTPException, Depends
from pathlib import Path
import time
import json
from typing import Optional
from vectorforge.core.indexing.hnsw_builder import build_hnsw_index_from_npz
try:
    from vectorforge.db import SessionLocal, IndexMeta as DBIndex
    DB_AVAILABLE = True
except Exception:
    SessionLocal = None
    DBIndex = None
    DB_AVAILABLE = False

router = APIRouter()
_INDEX_CACHE = {}


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@router.post("/collections/{collection_id}/indexes")
def build_index(collection_id: str, M: Optional[int] = 16, ef_construction: Optional[int] = 200, space: Optional[str] = "cosine"):
    base = Path("data") / "collections"
    npz_path = base / f"{collection_id}.npz"
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail="Collection embeddings not found on disk. Run ingest CLI first.")

    out_index = base / f"{collection_id}.index"
    meta_path = base / f"{collection_id}.index.json"

    start = time.time()
    try:
        build_hnsw_index_from_npz(str(npz_path), str(out_index), M=M, ef_construction=ef_construction, space=space)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    elapsed = int((time.time() - start) * 1000)

    meta = {
        "collection_id": collection_id,
        "index_path": str(out_index),
        "parameters": {"M": M, "ef_construction": ef_construction, "space": space},
        "build_time_ms": elapsed,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # persist index metadata to DB
    try:
        db = next(get_db())
        idx = DBIndex(id=collection_id, collection_id=collection_id, index_type="hnsw", index_path=str(out_index), parameters_json=json.dumps(meta["parameters"]), vector_count=None, dimension=None, build_time_ms=elapsed, status="built")
        db.add(idx)
        db.commit()
    except Exception:
        pass

    return {"index_id": f"{collection_id}", "status": "built", "metadata": meta}


@router.post("/collections/{collection_id}/indexes/load")
def load_index(collection_id: str):
    base = Path("data") / "collections"
    index_path = base / f"{collection_id}.index"
    meta_path = base / f"{collection_id}.index.json"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Index file not found")
    # load metadata if exists
    meta = None
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    # instantiate HNSW retriever using ids from npz
    npz_path = base / f"{collection_id}.npz"
    if not npz_path.exists():
        raise HTTPException(status_code=404, detail="Collection .npz not found; cannot load index without ids")
    import numpy as np
    data = np.load(str(npz_path), allow_pickle=True)
    ids = data.get("ids", None)
    if ids is None:
        raise HTTPException(status_code=500, detail=".npz missing ids")
    from vectorforge.core.retrieval.hnsw import HNSWRetriever
    dim = int(data["vectors"].shape[1])
    hr = HNSWRetriever(dim=dim, space=(meta or {}).get("parameters", {}).get("space", "cosine"))
    hr.load(str(index_path), ids.tolist())
    _INDEX_CACHE[collection_id] = hr
    # also persist index load status
    try:
        if DB_AVAILABLE:
            db = next(get_db())
            if db is not None:
                existing = db.query(DBIndex).filter(DBIndex.id == collection_id).first()
                if existing:
                    existing.status = "loaded"
                    db.commit()
    except Exception:
        pass
    # also populate search route's index retrievers cache if search module imported
    try:
        from . import search as search_mod
        search_mod._INDEX_RETRIEVERS[collection_id] = hr
    except Exception:
        pass
    return {"collection_id": collection_id, "status": "index_loaded", "metadata": meta}


@router.post("/collections/{collection_id}/indexes/unload")
def unload_index(collection_id: str):
    if collection_id in _INDEX_CACHE:
        del _INDEX_CACHE[collection_id]
    return {"collection_id": collection_id, "status": "index_unloaded"}
