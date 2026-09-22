# VectorForge

VectorForge is a modular retrieval engine and experimental platform for studying exact vs approximate (HNSW) and hybrid retrieval trade-offs.

Features
--------
- Exact nearest-neighbor search (NumPy)
- HNSW retrieval (`hnswlib`, optional)
- BM25 lexical retrieval and Hybrid lexical+dense fusion
- Evaluation metrics: Recall@K, MRR, nDCG
- Benchmarking package with reproducible experiment artifacts and provenance capture

Quick Start
-----------
1. Install optional dependencies for full benchmarking (recommended):

```bash
pip install -r requirements-dev.txt
# or selectively: pip install sentence-transformers hnswlib psutil pandas matplotlib
```

2. Generate a sample collection (10k documents):

```bash
python -m vectorforge.scripts.generate_sample_collection --n 10000 --prefix exp10k
```

3. Run Experiment A (Exact vs HNSW) which captures provenance and writes results to `results/`:

```bash
python -m vectorforge.scripts.experiment_a
```

4. Aggregate a numeric summary from a saved experiment JSON:

```bash
python -m vectorforge.scripts.aggregate_experiment results/<experiment_id>.json
```

Experiment & Provenance
------------------------
Experiments save JSON and summary CSVs in `results/`. For reproducibility we capture:

- `experiment_id`, `timestamp`, `git_commit`
- dataset & query SHA256 hashes
- `embedding_model` and `embedding_dimension`
- `top_k`, HNSW params (`M`, `ef_construction`, `ef_search`), `alpha` for hybrid
- warmup/measurement counts and timing
- environment artifacts: `results/pip_freeze.txt`, `results/hardware.txt`

Scripts
-------
- `vectorforge/scripts/experiment_a.py` — runs Experiment A (exact vs hnsw) and writes augmented results with provenance.
- `vectorforge/scripts/aggregate_experiment.py` — prints numeric summaries (Recall@K, MRR, nDCG, latency percentiles, QPS, RSS).
- `vectorforge/scripts/run_experiments.py` — convenience orchestrator for baseline + sweeps + dense/hybrid runs.
- `vectorforge/scripts/plot_results.py` — quick plotting utilities to generate PNGs from consolidated CSVs.

Notes
-----
- `hnswlib` is optional; on Windows prefer installing via conda to avoid C build tool issues: `conda install -c conda-forge hnswlib`.
- `sentence-transformers` is used for dense embeddings (`all-MiniLM-L6-v2` by default); if not installed a deterministic fallback exists for tests but is disabled by default for research runs.

Contributing
------------
Run tests with `pytest`. Add small deterministic tests for benchmarking components when changing metrics or runner logic.

License
-------
MIT
