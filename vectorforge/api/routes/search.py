from fastapi import APIRouter, HTTPException
from ..schemas.search import SearchRequest, SearchResponse, SearchResultItem
import time
import numpy as np
from vectorforge.core.retrieval.exact import ExactRetriever
from vectorforge.core.retrieval.hnsw import HNSWRetriever
from vectorforge.core.retrieval.bm25 import BM25Retriever
from vectorforge.core.retrieval.hybrid import HybridRetriever
from pathlib import Path

router = APIRouter()

# For this minimal implementation we'll load index files or rely on in-memory datasets produced by CLI
_EMBEDDINGS = {}  # collection_id -> {vectors, ids, texts}
_HNSW_INDEXES = {}  # collection_id -> HNSW path
_INDEX_RETRIEVERS = {}  # collection_id -> HNSWRetriever instance (loaded via indexes router)


def _load_collection_npz_if_present(collection_id: str):
    if collection_id in _EMBEDDINGS:
        return True
    base = Path("data") / "collections"
    npz_path = base / f"{collection_id}.npz"
    if not npz_path.exists():
        return False
    try:
        arr = np.load(str(npz_path), allow_pickle=True)
        vectors = arr["vectors"]
        ids = arr["ids"].tolist()
        texts = arr["texts"].tolist()
        _EMBEDDINGS[collection_id] = {"vectors": vectors, "ids": np.array(ids), "texts": np.array(texts)}
        return True
    except Exception:
        return False


@router.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    start = time.time()
    coll = _EMBEDDINGS.get(req.collection_id)
    if coll is None:
        loaded = _load_collection_npz_if_present(req.collection_id)
        if not loaded:
            raise HTTPException(status_code=404, detail="Collection embeddings not found in memory or on disk. Run ingest CLI first.")
        coll = _EMBEDDINGS.get(req.collection_id)

    # prepare metadata to return
    metadata = {"collection_id": req.collection_id}

    if req.method == "exact":
        er = ExactRetriever(np.array(coll["vectors"]), coll["ids"].tolist())
        # embed query with fallback random provider
        from vectorforge.core.embeddings.provider import EmbeddingProvider
        ep = EmbeddingProvider()
        qv = ep.embed(req.query)
        res = er.search(qv, top_k=req.top_k)
        items = [SearchResultItem(chunk_id=r["id"], document_id=None, score=r["score"], text=None) for r in res]
    elif req.method == "hnsw":
        # prefer using loaded index retriever if available
        idx_ret = _INDEX_RETRIEVERS.get(req.collection_id)
        if idx_ret is not None:
            from vectorforge.core.embeddings.provider import EmbeddingProvider
            ep = EmbeddingProvider()
            qv = ep.embed(req.query)
            res = idx_ret.search(qv, top_k=req.top_k)
            items = [SearchResultItem(chunk_id=r["id"], document_id=None, score=r["score"], text=None) for r in res]
            metadata["index_used"] = True
        else:
            # build a temporary HNSW retriever from vectors
            er = HNSWRetriever(dim=int(np.array(coll["vectors"]).shape[1]))
            er.build(np.array(coll["vectors"]), coll["ids"].tolist())
            from vectorforge.core.embeddings.provider import EmbeddingProvider
            ep = EmbeddingProvider()
            qv = ep.embed(req.query)
            res = er.search(qv, top_k=req.top_k)
            items = [SearchResultItem(chunk_id=r["id"], document_id=None, score=r["score"], text=None) for r in res]
            metadata["index_used"] = False
    elif req.method == "bm25":
        br = BM25Retriever()
        br.build(coll["texts"].tolist(), coll["ids"].tolist())
        res = br.search(req.query, top_k=req.top_k)
        items = [SearchResultItem(chunk_id=r["id"], document_id=None, score=r["score"], text=None) for r in res]
    elif req.method == "hybrid":
        er = ExactRetriever(np.array(coll["vectors"]), coll["ids"].tolist())
        br = BM25Retriever()
        br.build(coll["texts"].tolist(), coll["ids"].tolist())
        hybrid = HybridRetriever(er, br, alpha=(req.parameters or {}).get("alpha", 0.5))
        from vectorforge.core.embeddings.provider import EmbeddingProvider
        ep = EmbeddingProvider()
        qv = ep.embed(req.query)
        res = hybrid.search(req.query, qv, top_k=req.top_k)
        items = [SearchResultItem(chunk_id=r["id"], document_id=None, score=r["score"], text=None) for r in res]
    else:
        raise HTTPException(status_code=400, detail="Unsupported method")

    elapsed = (time.time() - start) * 1000.0
    return SearchResponse(query=req.query, method=req.method, latency_ms=elapsed, results=items, metadata=metadata)
