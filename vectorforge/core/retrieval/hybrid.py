from __future__ import annotations
from typing import List, Dict, Any
import numpy as np

from .base import Retriever


class HybridRetriever(Retriever):
    """Combine dense and BM25 scores with linear fusion.

    Usage:
        hybrid = HybridRetriever(dense_retriever, bm25_retriever, alpha=0.5)
        results = hybrid.search(query_text, query_vector, top_k=10,
                                dense_k=50, bm25_k=50)
    """

    def __init__(self, dense_retriever: Retriever, bm25_retriever: Retriever, alpha: float = 0.5):
        assert 0.0 <= alpha <= 1.0
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.alpha = alpha

    @staticmethod
    def _normalize_scores(scores: List[float]) -> List[float]:
        if not scores:
            return []
        arr = np.array(scores, dtype=float)
        lo = arr.min()
        hi = arr.max()
        if hi - lo <= 1e-12:
            return [1.0 for _ in arr]
        return ((arr - lo) / (hi - lo)).tolist()

    def search(self, query_text: str, query_vector: Any, top_k: int = 10, dense_k: int = 50, bm25_k: int = 50) -> List[Dict[str, Any]]:
        # retrieve from dense and bm25
        dense_res = self.dense.search(query_vector, top_k=dense_k)
        bm25_res = self.bm25.search(query_text, top_k=bm25_k)

        dense_ids = [r["id"] for r in dense_res]
        dense_scores = [r["score"] for r in dense_res]
        bm25_ids = [r["id"] for r in bm25_res]
        bm25_scores = [r["score"] for r in bm25_res]

        norm_dense = dict(zip(dense_ids, self._normalize_scores(dense_scores)))
        norm_bm25 = dict(zip(bm25_ids, self._normalize_scores(bm25_scores)))

        # union of candidates
        all_ids = list(dict.fromkeys(dense_ids + bm25_ids))
        combined = []
        for cid in all_ids:
            ds = norm_dense.get(cid, 0.0)
            bs = norm_bm25.get(cid, 0.0)
            fused = self.alpha * ds + (1.0 - self.alpha) * bs
            combined.append({"id": cid, "score": float(fused), "dense_score": float(ds), "bm25_score": float(bs)})

        combined_sorted = sorted(combined, key=lambda x: -x["score"])[:top_k]
        return combined_sorted
