from __future__ import annotations
from typing import List, Dict, Any


class Retriever:
    """Abstract retriever interface."""

    def build(self, *args, **kwargs) -> None:
        raise NotImplementedError()

    def search(self, query_vector, top_k: int = 10) -> List[Dict[str, Any]]:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        raise NotImplementedError()

    def load(self, path: str) -> None:
        raise NotImplementedError()

    def stats(self) -> Dict[str, Any]:
        return {}
