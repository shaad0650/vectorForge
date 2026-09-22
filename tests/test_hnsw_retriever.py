import numpy as np
import pytest

try:
    import hnswlib
except Exception:
    hnswlib = None

from vectorforge.core.retrieval.hnsw import HNSWRetriever


@pytest.mark.skipif(hnswlib is None, reason="hnswlib not installed")
def test_hnsw_retriever():
    vecs = np.random.RandomState(0).randn(50, 16).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)
    ids = [f"d{i}" for i in range(50)]
    h = HNSWRetriever(dim=16)
    h.build(vecs, ids, M=8, ef_construction=100)
    q = vecs[0]
    res = h.search(q, top_k=5, ef_search=50)
    assert len(res) == 5
