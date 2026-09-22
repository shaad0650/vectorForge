from __future__ import annotations
from typing import List, Dict, Any
from vectorforge.core.evaluation.metrics import recall_at_k, mrr, ndcg_at_k


class MetricsEvaluator:
    def __init__(self, top_ks: List[int]):
        self.top_ks = top_ks

    def evaluate(self, retrieved_ids: List[str], relevant_ids: List[str]) -> Dict[str, float]:
        out = {}
        for k in self.top_ks:
            out[f"recall@{k}"] = recall_at_k(retrieved_ids, relevant_ids, k)
            out[f"ndcg@{k}"] = ndcg_at_k(retrieved_ids, relevant_ids, k)
        out["mrr"] = mrr(retrieved_ids, relevant_ids)
        return out
