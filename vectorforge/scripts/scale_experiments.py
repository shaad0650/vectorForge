"""Orchestrator to run scale experiments (10k, 50k, 100k) by generating datasets and invoking run_benchmark.
This script writes configs to `results/` and calls the existing runner.
"""
from __future__ import annotations
import subprocess
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)

def gen_and_run(n):
    prefix = f"scale_{n}"
    print('generating', n)
    subprocess.check_call(["python", "-m", "vectorforge.scripts.generate_sample_collection", "--n", str(n), "--prefix", prefix])
    npz = f"data/collections/{prefix}_bench.npz"
    queries = f"data/collections/{prefix}_queries.jsonl"
    cfg = RESULTS / f"cfg_scale_{n}_{uuid.uuid4().hex}.yml"
    cfg.write_text(f"""
name: scale_{n}
dataset:
  path: {npz}
queries:
  path: {queries}
methods:
  - exact
  - hnsw
retrieval:
  top_k: [1,5,10]
measurement:
  measurement_queries: 200
embeddings:
  allow_fallback: false
sweep: {{}}
""")
    subprocess.check_call(["python", "-m", "vectorforge.scripts.run_benchmark", str(cfg)])

def main():
    for n in (10000, 50000, 100000):
        gen_and_run(n)

if __name__ == '__main__':
    main()
