from __future__ import annotations
from typing import List, Dict, Any
import numpy as np
from ..retrieval.base import Retriever


class ExactRetriever(Retriever):
    """Brute-force exact nearest-neighbour search using NumPy.

    Stores vectors as a (N, D) NumPy array and returns top-K by cosine similarity.
    """

    def __init__(self, vectors: np.ndarray, ids: List[str]):
        assert vectors.ndim == 2
        assert vectors.shape[0] == len(ids)
        self.vectors = vectors.astype(np.float32)
        self.ids = ids
        # precompute norms
        norms = np.linalg.norm(self.vectors, axis=1)
        norms[norms == 0.0] = 1.0
        self.norms = norms

    def save(self, path: str) -> None:
        # save vectors and ids to a npz
        np.savez_compressed(path, vectors=self.vectors, ids=np.array(self.ids, dtype=object))

    @classmethod
    def load(cls, path: str) -> "ExactRetriever":
        data = np.load(path, allow_pickle=True)
        vecs = data["vectors"]
        ids = data["ids"].tolist()
        return cls(vecs, ids)

    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Dict[str, Any]]:
        q = np.array(query_vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            q_norm = 1.0
        sims = (self.vectors @ q) / (self.norms * q_norm)
        idx = np.argsort(-sims)[:top_k]
        results = []
        for i in idx:
            results.append({"id": self.ids[i], "score": float(sims[i])})
        return results
