"""Orchestrator that runs a sequence of experiments: baseline exact vs hnsw, hnsw sweep, and dense/bm25/hybrid comparisons.

This is a convenience script for local experiments. It uses the existing benchmark runner and config format.
"""
from __future__ import annotations
import subprocess
import json
from pathlib import Path
import uuid

# Simple helper to write temporary bench configs and call the CLI
TEMPLATE = """
name: {name}
dataset:
  path: {npz}
queries:
  path: {queries}
methods:
{methods}
retrieval:
  top_k: [1,5,10]
hnsw:
  M: {M}
  ef_construction: {efc}
  ef_search: {efs}
measurement:
  measurement_queries: {measurement_queries}
embeddings:
  allow_fallback: false
sweep: {sweep}
"""


def _methods_block(meths):
    return "\n".join([f"  - {m}" for m in meths])


def run_cli(config_path: Path):
    subprocess.check_call(["python", "-m", "vectorforge.scripts.run_benchmark", str(config_path)])


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)

    # dataset: generate 10k small synthetic dataset locally
    print("Generating 10k dataset (this may take a moment)...")
    subprocess.check_call(["python", "-m", "vectorforge.scripts.generate_sample_collection", "--n", "10000", "--prefix", "exp10k"])
    npz = Path("data/collections/exp10k_bench.npz").resolve()
    queries = Path("data/collections/exp10k_queries.jsonl").resolve()

    # 1. Baseline: exact vs hnsw
    cfg1 = TEMPLATE.format(
        name="baseline_exact_hnsw",
        npz=npz.as_posix(),
        queries=queries.as_posix(),
        methods=_methods_block(["exact", "hnsw"]),
        M=16,
        efc=200,
        efs=100,
        measurement_queries=200,
        sweep="{}",
    )
    cfg_path = out / f"cfg_baseline_{uuid.uuid4().hex}.yml"
    cfg_path.write_text(cfg1)
    run_cli(cfg_path)

    # 2. HNSW sweep (smaller sweep for demo):
    sweep_spec = json.dumps({
        "hnsw": {"M": [8, 16, 32], "ef_construction": [100, 200], "ef_search": [50, 100, 200]},
        "hybrid": {"alpha": [0.25, 0.5, 0.75]},
    })
    cfg2 = TEMPLATE.format(
        name="sweep_hnsw",
        npz=npz.as_posix(),
        queries=queries.as_posix(),
        methods=_methods_block(["hnsw"]),
        M=16,
        efc=200,
        efs=100,
        measurement_queries=200,
        sweep=sweep_spec,
    )
    cfg_path2 = out / f"cfg_hnsw_sweep_{uuid.uuid4().hex}.yml"
    cfg_path2.write_text(cfg2)
    run_cli(cfg_path2)

    # 3. Dense vs BM25 vs Hybrid
    cfg3 = TEMPLATE.format(
        name="dense_bm25_hybrid",
        npz=npz.as_posix(),
        queries=queries.as_posix(),
        methods=_methods_block(["bm25", "exact", "hybrid"]),
        M=16,
        efc=200,
        efs=100,
        measurement_queries=200,
        sweep=json.dumps({"hybrid": {"alpha": [0.25, 0.5, 0.75]}}),
    )
    cfg_path3 = out / f"cfg_dense_bm25_{uuid.uuid4().hex}.yml"
    cfg_path3.write_text(cfg3)
    run_cli(cfg_path3)

    print("Experiments finished. Check results/ for summaries.")


if __name__ == "__main__":
    main()
