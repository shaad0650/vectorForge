"""Generate a small sample collection `.npz` and queries `.jsonl` for smoke benchmarking."""
from __future__ import annotations
import numpy as np
import json
from pathlib import Path


def main(out_dir: str = "data/collections", n: int = 50, dim: int = 384, prefix: str = "itest"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(12345)
    vectors = rng.standard_normal((n, dim)).astype(np.float32)
    ids = [f"doc_{i}" for i in range(n)]
    texts = [f"This is the text of document {i}. It mentions topic {(i%5)}." for i in range(n)]

    np.savez(out / f"{prefix}_bench.npz", vectors=vectors, ids=np.array(ids, dtype=object), texts=np.array(texts, dtype=object))

    # create simple queries targeting a few docs
    queries = []
    for qid in range(100):
        qtext = f"documents about topic {qid % 20}"
        # relevant ids are those with matching topic
        relevant = [f"doc_{i}" for i in range(n) if (i % 20) == (qid % 20)][:10]
        queries.append({"query": qtext, "relevant": relevant})

    with open(out / f"{prefix}_queries.jsonl", "w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")

    print("Wrote:", out / f"{prefix}_bench.npz", out / f"{prefix}_queries.jsonl")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=50, help="number of documents to generate")
    p.add_argument("--dim", type=int, default=384, help="embedding dimension")
    p.add_argument("--out", default="data/collections", help="output directory")
    p.add_argument("--prefix", default="itest", help="file prefix")
    args = p.parse_args()
    main(out_dir=args.out, n=args.n, dim=args.dim, prefix=args.prefix)
