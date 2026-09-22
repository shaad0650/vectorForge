from __future__ import annotations
from typing import List
import math


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if len(relevant) == 0:
        return 0.0
    retrieved_k = retrieved[:k]
    rel = set(relevant)
    return len([r for r in retrieved_k if r in rel]) / len(rel)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    rel = set(relevant)
    for i, r in enumerate(retrieved, start=1):
        if r in rel:
            return 1.0 / i
    return 0.0


def dcg_at_k(retrieved: List[str], gains: List[float], k: int) -> float:
    dcg = 0.0
    # protect against gains being shorter than retrieved (use min length)
    for i in range(min(k, len(retrieved), len(gains))):
        dcg += gains[i] / math.log2(i + 2)
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    # relevant is treated as list in decreasing relevance; for now binary relevance
    ideal_gains = [1.0 for _ in range(min(len(relevant), k))]
    gains = [1.0 if r in set(relevant) else 0.0 for r in retrieved[:k]]
    idcg = dcg_at_k(retrieved, ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg_at_k(retrieved, gains, k) / idcg
