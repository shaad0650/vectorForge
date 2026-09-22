from __future__ import annotations
from typing import List
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None


class EmbeddingProvider:
    """Simple embedding provider. Uses SentenceTransformers if available, otherwise uses random vectors (for tests)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        if SentenceTransformer is not None:
            self.model = SentenceTransformer(model_name)

    def embed(self, text: str) -> List[float]:
        if self.model is not None:
            v = self.model.encode(text, convert_to_numpy=True)
            return v.tolist()
        # fallback: deterministic random based on hash
        rng = np.random.default_rng(abs(hash(text)) % (2**32))
        return rng.standard_normal(384).astype(float).tolist()

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if self.model is not None:
            arr = self.model.encode(texts, convert_to_numpy=True)
            return arr.tolist()
        return [self.embed(t) for t in texts]

    def dimension(self) -> int:
        if self.model is not None:
            return self.model.get_sentence_embedding_dimension()
        return 384
