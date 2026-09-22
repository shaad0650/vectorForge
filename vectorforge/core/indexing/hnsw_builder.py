from __future__ import annotations
import numpy as np
from typing import Optional

try:
    import hnswlib
except Exception:
    hnswlib = None


def build_hnsw_index_from_npz(npz_path: str, out_index_path: str, M: int = 16, ef_construction: int = 200, space: str = "cosine") -> None:
    if hnswlib is None:
        raise RuntimeError("hnswlib is required to build HNSW indexes")
    data = np.load(npz_path, allow_pickle=True)
    vecs = data["vectors"]
    ids = data.get("ids", None)
    n, dim = vecs.shape
    p = hnswlib.Index(space=space, dim=dim)
    p.init_index(max_elements=n, ef_construction=ef_construction, M=M)
    p.add_items(vecs, list(range(n)))
    p.save_index(out_index_path)
