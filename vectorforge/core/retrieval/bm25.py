from __future__ import annotations
from typing import List, Dict, Any, Optional
import numpy as np

try:
    from rank_bm25 import BM25Okapi
except Exception:
    BM25Okapi = None

from ..retrieval.base import Retriever

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
except Exception:
    TfidfVectorizer = None
    linear_kernel = None


class BM25Retriever(Retriever):
    """BM25 lexical retriever. Uses `rank_bm25` if available, otherwise falls back to TF-IDF cosine scoring.

    Provide `build(docs)` then `search(query, top_k)` returning list of {id, score}.
    """

    def __init__(self):
        self.bm25 = None
        self.tfidf = None
        self.doc_ids: List[str] = []
        self.docs: List[str] = []

    def build(self, docs: List[str], ids: List[str]):
        self.doc_ids = ids
        self.docs = docs
        if BM25Okapi is not None:
            tokenized = [d.split() for d in docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            if TfidfVectorizer is None:
                raise RuntimeError("scikit-learn is required for TF-IDF fallback but it's not installed")
            self.tfidf = TfidfVectorizer().fit_transform(docs)

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        if self.bm25 is not None:
            qtok = query.split()
            scores = self.bm25.get_scores(qtok)
            idx = np.argsort(-scores)[:top_k]
            return [{"id": self.doc_ids[i], "score": float(scores[i])} for i in idx]
        elif self.tfidf is not None:
            if TfidfVectorizer is None:
                raise RuntimeError("scikit-learn is required for TF-IDF fallback but it's not installed")
            qvec = TfidfVectorizer().fit(self.docs).transform([query])
            sims = linear_kernel(qvec, self.tfidf).flatten()
            idx = np.argsort(-sims)[:top_k]
            return [{"id": self.doc_ids[i], "score": float(sims[i])} for i in idx]
        else:
            raise RuntimeError("BM25Retriever.build must be called before search")
