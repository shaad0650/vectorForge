import numpy as np
from vectorforge.core.retrieval.exact import ExactRetriever
import os


def test_exact_save_load(tmp_path):
    vecs = np.random.RandomState(0).random((10, 8)).astype('float32')
    ids = [f"d{i}" for i in range(10)]
    er = ExactRetriever(vecs, ids)
    p = tmp_path / "test.npz"
    er.save(str(p))
    er2 = ExactRetriever.load(str(p))
    assert er2.ids == ids
    assert er2.vectors.shape == vecs.shape
