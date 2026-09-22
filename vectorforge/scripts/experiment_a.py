"""Run Experiment A: Exact vs HNSW with provenance and artifact capture.

Generates dataset if missing, runs a controlled benchmark comparing `exact` and `hnsw`,
captures provenance (git commit, dataset/query hashes, embedding model, seed, hardware),
and augments the JSON result written by the benchmark runner.
"""
from __future__ import annotations
import hashlib
import json
import os
import subprocess
import platform
from pathlib import Path
from datetime import datetime

from vectorforge.benchmark.config import BenchmarkConfig, HNSWConfig
from vectorforge.benchmark.runner import BenchmarkRunner
from vectorforge.core.embeddings.provider import EmbeddingProvider


def sha256_of_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit_hash() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
        return out.decode().strip()
    except Exception:
        return None


def ensure_dataset(n=10000, prefix="exp10k") -> tuple[Path, Path]:
    npz = Path(f"data/collections/{prefix}_bench.npz")
    q = Path(f"data/collections/{prefix}_queries.jsonl")
    if not npz.exists() or not q.exists():
        print("Generating dataset...")
        subprocess.check_call(["python", "-m", "vectorforge.scripts.generate_sample_collection", "--n", str(n), "--prefix", prefix])
    return npz.resolve(), q.resolve()


def capture_pip_freeze(out_dir: Path):
    try:
        with open(out_dir / "pip_freeze.txt", "w", encoding="utf-8") as f:
            subprocess.check_call(["python", "-m", "pip", "freeze"], stdout=f)
    except Exception:
        pass


def write_hardware(out_dir: Path):
    try:
        import psutil
    except Exception:
        psutil = None
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
    }
    if psutil:
        info.update({
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "total_ram_bytes": psutil.virtual_memory().total,
        })
    with open(out_dir / "hardware.txt", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


def augment_results(exp_id: str, out_dir: Path, provenance: dict):
    p = out_dir / f"{exp_id}.json"
    if not p.exists():
        print("Result JSON not found to augment:", p)
        return
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    wrapped = {
        "provenance": provenance,
        "results": data,
    }
    with open(out_dir / f"{exp_id}_with_provenance.json", "w", encoding="utf-8") as f:
        json.dump(wrapped, f, indent=2)
    print("Wrote augmented results:", out_dir / f"{exp_id}_with_provenance.json")


def main():
    out = Path("results")
    out.mkdir(exist_ok=True)

    npz, queries = ensure_dataset(n=10000, prefix="exp10k")

    # Controlled config for Experiment A
    cfg = BenchmarkConfig(
        name="experiment_a_exact_vs_hnsw",
        dataset_path=str(npz),
        queries_path=str(queries),
        methods=["exact", "hnsw"],
        top_k=[1, 5, 10],
        hnsw=HNSWConfig(M=16, ef_construction=200, ef_search=100),
        hybrid_alpha=[0.5],
        queries_limit=1000,
        allow_fallback_embeddings=False,
        warmup_queries=50,
        measurement_queries=200,
    )

    runner = BenchmarkRunner(cfg)
    outobj = runner.run()
    exp_id = outobj.get("experiment_id")

    # provenance
    ep = EmbeddingProvider()
    prov = {
        "experiment_id": exp_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "git_commit": git_commit_hash(),
        "dataset": str(npz),
        "dataset_hash": sha256_of_file(npz) if npz.exists() else None,
        "query_set": str(queries),
        "query_set_hash": sha256_of_file(queries) if queries.exists() else None,
        "embedding_model": ep.model_name,
        "embedding_dimension": ep.dimension(),
        "top_k": cfg.top_k,
        "random_seed": None,
        "method_list": cfg.methods,
        "hnsw": {"M": cfg.hnsw.M, "ef_construction": cfg.hnsw.ef_construction, "ef_search": cfg.hnsw.ef_search},
        "warmup_queries": cfg.warmup_queries,
        "measurement_queries": cfg.measurement_queries,
    }

    # capture environment artifacts
    capture_pip_freeze(out)
    write_hardware(out)

    augment_results(exp_id, out, prov)

    print("Experiment A complete. Experiment id:", exp_id)


if __name__ == "__main__":
    main()
