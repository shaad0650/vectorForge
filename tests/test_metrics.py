from vectorforge.core.evaluation.metrics import recall_at_k, mrr, ndcg_at_k


def test_recall_mrr_ndcg():
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d"]

    assert recall_at_k(retrieved, relevant, 1) == 0.0
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert mrr(retrieved, relevant) == 0.5

    ndcg = ndcg_at_k(retrieved, relevant, 2)
    assert ndcg > 0.0
