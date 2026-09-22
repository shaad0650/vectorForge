from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np
try:
    import hnswlib
except Exception:
    hnswlib = None

from ..retrieval.base import Retriever


class HNSWRetriever(Retriever):
    def __init__(self, dim: int, space: str = "cosine"):
        self.dim = dim
        self.space = space
        self.index: Optional[hnswlib.Index] = None
        self.id_map: List[str] = []

    def build(self, vectors: np.ndarray, ids: List[str], M: int = 16, ef_construction: int = 200):
        if hnswlib is None:
            raise RuntimeError("hnswlib is required for HNSWRetriever but it's not installed")
        n, d = vectors.shape
        assert d == self.dim
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.init_index(max_elements=n, ef_construction=ef_construction, M=M)
        self.index.add_items(vectors, list(range(n)))
        self.index.set_ef(50)
        self.id_map = ids

    def search(self, query_vector: np.ndarray, top_k: int = 10, ef_search: int = 50) -> List[Dict[str, Any]]:
        if hnswlib is None:
            raise RuntimeError("hnswlib is required for HNSWRetriever but it's not installed")
        assert self.index is not None
        self.index.set_ef(ef_search)
        labels, distances = self.index.knn_query(query_vector, k=top_k)
        res = []
        for label, dist in zip(labels[0], distances[0]):
            rid = self.id_map[label]
            # hnswlib returns distance; convert to similarity for cosine if needed
            score = float(1.0 - dist) if self.space == "cosine" else float(-dist)
            res.append({"id": rid, "score": score})
        return res

    def save(self, path: str) -> None:
        if self.index:
            self.index.save_index(path)

    def load(self, path: str, ids: List[str]) -> None:
        self.index = hnswlib.Index(space=self.space, dim=self.dim)
        self.index.load_index(path)
        self.id_map = ids
