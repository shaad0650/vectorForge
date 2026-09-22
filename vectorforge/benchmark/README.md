# VectorForge Benchmarking

Quick instructions for running the built-in benchmark runner.

Preparation
- Ensure you have a collection `.npz` with keys: `vectors` (ndarray), `ids` (array-like), and optionally `texts`.
- Prepare a queries file (`.jsonl`) where each line is JSON with keys: `query` (string) and `relevant` (list of ids).

Example `bench.yml` fields:

```yaml
name: quick_test
dataset:
  path: data/collections/your_collection.npz
queries:
  path: data/collections/queries.jsonl
methods:
  - exact
  - hnsw
retrieval:
  top_k: [5, 10]
hnsw:
  M: 16
  ef_construction: 100
  ef_search: 50
embeddings:
  allow_fallback: false  # set true for quick smoke tests using deterministic fallback
```

Run a benchmark (example):

```bash
python -m vectorforge.scripts.run_benchmark bench.yml
```

Outputs are written to `results/<experiment_id>.json` and `<experiment_id>_summary.csv`.

Notes
- For research-grade experiments, install `sentence-transformers` and configure `EmbeddingProvider` to use a real model; set `embeddings.allow_fallback: false` to prevent accidental fallback use.
- The runner rejects experiments that would use the deterministic fallback unless explicitly allowed by `allow_fallback`.
