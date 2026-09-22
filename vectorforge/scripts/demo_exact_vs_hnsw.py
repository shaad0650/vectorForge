"""Demo script: create random dataset, run exact and HNSW, compare metrics."""
import numpy as np
from vectorforge.core.retrieval.exact import ExactRetriever
from vectorforge.core.retrieval.hnsw import HNSWRetriever
from vectorforge.core.evaluation.metrics import recall_at_k, mrr, ndcg_at_k


def make_random_data(n=1000, dim=128, seed=42):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype(np.float32)
    # normalize for cosine
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs = vecs / np.maximum(norms, 1e-9)
    ids = [f"doc_{i}" for i in range(n)]
    return vecs, ids


def main():
    vecs, ids = make_random_data(n=500, dim=64)
    exact = ExactRetriever(vecs, ids)

    hnsw = HNSWRetriever(dim=64)
    hnsw.build(vecs, ids, M=8, ef_construction=100)

    # generate queries
    qvecs = vecs[:50]
    ks = [1, 5, 10]

    for i, q in enumerate(qvecs):
        exact_res = exact.search(q, top_k=50)
        exact_ids = [r["id"] for r in exact_res]

        hnsw_res = hnsw.search(q, top_k=50, ef_search=50)
        hnsw_ids = [r["id"] for r in hnsw_res]

        if i == 0:
            print("Query, Recall@1, Recall@5, Recall@10, MRR, nDCG@10")

        r1 = recall_at_k(hnsw_ids, exact_ids[:10], 1)
        r5 = recall_at_k(hnsw_ids, exact_ids[:10], 5)
        r10 = recall_at_k(hnsw_ids, exact_ids[:10], 10)
        _mrr = mrr(hnsw_ids, exact_ids[:10])
        _ndcg = ndcg_at_k(hnsw_ids, exact_ids[:10], 10)
        print(f"q{i},{r1:.3f},{r5:.3f},{r10:.3f},{_mrr:.3f},{_ndcg:.3f}")


if __name__ == "__main__":
    main()
