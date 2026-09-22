import numpy as np
from vectorforge.core.retrieval.exact import ExactRetriever
from vectorforge.core.retrieval.bm25 import BM25Retriever
from vectorforge.core.retrieval.hybrid import HybridRetriever


def test_hybrid_basic():
    docs = ["alpha beta gamma", "delta epsilon zeta", "alpha delta"]
    ids = ["d1", "d2", "d3"]
    # create simple dense vectors
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.2]], dtype=float)
    # normalize
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)

    exact = ExactRetriever(vecs, ids)
    bm25 = BM25Retriever()
    bm25.build(docs, ids)

    hybrid = HybridRetriever(exact, bm25, alpha=0.6)
    qvec = vecs[0]
    res = hybrid.search("alpha gamma", qvec, top_k=2)
    assert len(res) == 2
