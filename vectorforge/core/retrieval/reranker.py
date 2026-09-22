from __future__ import annotations
from typing import List, Dict, Any


class Reranker:
    """Simple reranker scaffold. Plug in a model later.

    `score(candidate_text, query_text)` should return a float score (higher is better).
    """

    def __init__(self):
        pass

    def score(self, candidate_text: str, query_text: str) -> float:
        # placeholder: identity heuristic
        return 0.0

    def rerank(self, candidates: List[Dict[str, Any]], texts: Dict[str, str], query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        scored = []
        for c in candidates:
            cid = c["id"]
            s = self.score(texts.get(cid, ""), query_text)
            scored.append({**c, "rerank_score": float(s)})
        scored_sorted = sorted(scored, key=lambda x: -x["rerank_score"])[:top_k]
        return scored_sorted
