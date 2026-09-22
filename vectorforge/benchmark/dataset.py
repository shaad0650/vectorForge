from __future__ import annotations
from typing import List, Dict, Any
import numpy as np


class BenchmarkDataset:
    """Loads dataset from an .npz file with keys: vectors, ids, texts"""

    def __init__(self, npz_path: str):
        self.npz_path = npz_path
        data = np.load(npz_path, allow_pickle=True)
        self.vectors = data["vectors"]
        self.ids = data["ids"].tolist()
        self.texts = data.get("texts", None)

    def __len__(self):
        return self.vectors.shape[0]

    def get_vector(self, idx: int):
        return self.vectors[idx]

    def all_ids(self):
        return self.ids
