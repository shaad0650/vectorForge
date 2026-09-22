import numpy as np
from vectorforge.core.retrieval.exact import ExactRetriever


def test_exact_retriever():
    vecs = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    # normalize
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / norms
    ids = ["d1", "d2", "d3"]
    retriever = ExactRetriever(vecs, ids)
    q = np.array([1.0, 0.0], dtype=np.float32)
    res = retriever.search(q, top_k=2)
    assert res[0]["id"] == "d1"
