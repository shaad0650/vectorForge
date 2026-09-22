"""Aggregate and print numeric summary from a benchmark JSON.

Usage:
  python -m vectorforge.scripts.aggregate_experiment results/<experiment>.json
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
from statistics import mean


def summarize(results_path: Path):
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for exp in data.get("experiments", []):
        method = exp.get("method")
        print(f"\nMethod: {method}")
        lat = exp.get("latency_summary", {})
        print(" Latency summary:")
        print(f"  count: {lat.get('count')}")
        print(f"  mean_ms: {lat.get('mean_ms')}")
        print(f"  p50_ms: {lat.get('p50_ms')}")
        print(f"  p95_ms: {lat.get('p95_ms')}")
        print(f"  p99_ms: {lat.get('p99_ms')}")

        resources = exp.get("resources", {})
        print(" Resources:")
        print(f"  rss: {resources.get('rss')}")

        mq = exp.get("measurement_qps")
        if mq is not None:
            print(f" Measurement QPS: {mq:.2f}")

        # aggregate per-query metrics
        queries = exp.get("queries", [])
        if not queries:
            print(" No per-query metrics available")
            continue

        rec1 = []
        rec5 = []
        rec10 = []
        ndcg10 = []
        mrrs = []
        for q in queries:
            m = q.get("metrics", {})
            rec1.append(m.get("recall@1", 0.0))
            rec5.append(m.get("recall@5", 0.0))
            rec10.append(m.get("recall@10", 0.0))
            ndcg10.append(m.get("ndcg@10", 0.0))
            mrrs.append(m.get("mrr", 0.0))

        def fmt(x):
            return f"{x:.4f}" if x is not None else "N/A"

        print(" Quality (averaged over queries):")
        print(f"  recall@1: {fmt(mean(rec1))}")
        print(f"  recall@5: {fmt(mean(rec5))}")
        print(f"  recall@10: {fmt(mean(rec10))}")
        print(f"  mrr: {fmt(mean(mrrs))}")
        print(f"  ndcg@10: {fmt(mean(ndcg10))}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m vectorforge.scripts.aggregate_experiment path/to/experiment.json")
        sys.exit(1)
    p = Path(sys.argv[1])
    if not p.exists():
        print("File not found:", p)
        sys.exit(1)
    summarize(p)
