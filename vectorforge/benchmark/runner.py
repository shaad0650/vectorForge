from __future__ import annotations
from typing import List, Dict, Any
import time
import uuid
from .config import BenchmarkConfig
from .dataset import BenchmarkDataset
from .evaluator import MetricsEvaluator
from .writer import ResultWriter
from vectorforge.core.retrieval.exact import ExactRetriever
from vectorforge.core.retrieval.hnsw import HNSWRetriever
from vectorforge.core.retrieval.bm25 import BM25Retriever
from vectorforge.core.retrieval.hybrid import HybridRetriever
from vectorforge.core.embeddings.provider import EmbeddingProvider
from .monitor import ResourceMonitor
import numpy as np
import statistics


class BenchmarkRunner:
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.dataset = BenchmarkDataset(config.dataset_path)
        self.queries = self._load_queries(config.queries_path)
        self.evaluator = MetricsEvaluator(config.top_k)
        self.writer = ResultWriter()
        self.monitor = ResourceMonitor()

    def _load_queries(self, path: str):
        import json
        qs = []
        if path is None:
            return qs
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if self.config.queries_limit and i >= self.config.queries_limit:
                    break
                qs.append(json.loads(line))
        return qs

    def run(self) -> Dict[str, Any]:
        results = {"experiments": []}
        # build exact retriever once
        exact = ExactRetriever(np.array(self.dataset.vectors), self.dataset.ids)
        ep = EmbeddingProvider()
        # explicit fallback guard
        if not self.config.allow_fallback_embeddings and ep.model is None:
            raise RuntimeError(
                "EmbeddingProvider is using fallback deterministic embeddings. "
                "For reproducible research, configure a real embedding model or set "
                "`embeddings.allow_fallback: true` in your benchmark config. "
                "See vectorforge/benchmark/README.md for details."
            )

        # expand sweep parameters into experiment configurations
        sweep_configs = [
            {"method": m, "sweep_params": {}} for m in self.config.methods
        ]
        # currently support hnsw and hybrid sweeps
        if self.config.sweep:
            hcfg = self.config.sweep.get("hnsw", {})
            hybrid_cfg = self.config.sweep.get("hybrid", {})
            # if hnsw sweep present, expand entries for methods that are hnsw
            new_configs = []
            for sc in sweep_configs:
                if sc["method"] == "hnsw" and hcfg:
                    for M in hcfg.get("M", [self.config.hnsw.M]):
                        for efc in hcfg.get("ef_construction", [self.config.hnsw.ef_construction]):
                            for efs in hcfg.get("ef_search", [self.config.hnsw.ef_search]):
                                cfg = {"method": "hnsw", "sweep_params": {"M": M, "ef_construction": efc, "ef_search": efs}}
                                new_configs.append(cfg)
                elif sc["method"] == "hybrid" and hybrid_cfg:
                    for alpha in hybrid_cfg.get("alpha", self.config.hybrid_alpha):
                        cfg = {"method": "hybrid", "sweep_params": {"alpha": alpha}}
                        new_configs.append(cfg)
                else:
                    new_configs.append(sc)
            sweep_configs = new_configs

        for cfg in sweep_configs:
            method = cfg.get("method")
            sweep_params = cfg.get("sweep_params", {})
            # prepare method-level structures
            per_query_latencies = []
            method_results = []

            # some retrievers can be built once per method
            method_retriever = None
            if method == "hnsw":
                # allow sweep params to override M/ef_construction
                M = sweep_params.get("M", self.config.hnsw.M)
                efc = sweep_params.get("ef_construction", self.config.hnsw.ef_construction)
                try:
                    method_retriever = HNSWRetriever(dim=int(self.dataset.vectors.shape[1]))
                    method_retriever.build(np.array(self.dataset.vectors), self.dataset.ids, M=M, ef_construction=efc)
                except RuntimeError as e:
                    results["experiments"].append({"method": "hnsw", "skipped": True, "reason": str(e), "sweep_params": sweep_params})
                    continue
            elif method == "bm25":
                br = BM25Retriever()
                br.build(self.dataset.texts.tolist(), self.dataset.ids)
                method_retriever = br
            elif method == "hybrid":
                br = BM25Retriever()
                br.build(self.dataset.texts.tolist(), self.dataset.ids)
                method_retriever = HybridRetriever(exact, br, alpha=(self.config.hybrid_alpha[0] if self.config.hybrid_alpha else 0.5))

            for q in self.queries:
                query_text = q.get("query")
                relevant = q.get("relevant", [])
                qv = ep.embed(query_text)
                start = time.perf_counter()
                if method == "exact":
                    res = exact.search(qv, top_k=max(self.config.top_k))
                elif method == "hnsw":
                    ef_search = sweep_params.get("ef_search", self.config.hnsw.ef_search)
                    res = method_retriever.search(qv, top_k=max(self.config.top_k), ef_search=ef_search)
                elif method == "bm25":
                    res = method_retriever.search(query_text, top_k=max(self.config.top_k))
                elif method == "hybrid":
                    # hybrid sweep may change alpha
                    res = method_retriever.search(query_text, qv, top_k=max(self.config.top_k))
                else:
                    continue
                end = time.perf_counter()

                latency_ms = (end - start) * 1000.0
                per_query_latencies.append(latency_ms)

                retrieved_ids = [r["id"] for r in res]
                metrics = self.evaluator.evaluate(retrieved_ids, relevant)
                method_results.append({"query": query_text, "metrics": metrics, "latency_ms": latency_ms})

            # Warmup phase: run warmup queries or warmup_seconds without recording
            def _run_queries(n_queries=None, for_seconds=None):
                completed = 0
                start_time = time.perf_counter()
                i = 0
                while True:
                    if n_queries is not None and completed >= n_queries:
                        break
                    if for_seconds is not None and (time.perf_counter() - start_time) >= for_seconds:
                        break
                    q = self.queries[i % len(self.queries)]
                    qtext = q.get("query")
                    qv_local = ep.embed(qtext)
                    # execute but don't record latency
                    if method == "exact":
                        _ = exact.search(qv_local, top_k=max(self.config.top_k))
                    elif method == "hnsw":
                        ef_search = sweep_params.get("ef_search", self.config.hnsw.ef_search)
                        _ = method_retriever.search(qv_local, top_k=max(self.config.top_k), ef_search=ef_search)
                    elif method == "bm25":
                        _ = method_retriever.search(qtext, top_k=max(self.config.top_k))
                    elif method == "hybrid":
                        _ = method_retriever.search(qtext, qv_local, top_k=max(self.config.top_k))
                    completed += 1
                    i += 1
                return completed

            # run warmup if configured
            if self.config.warmup_queries or self.config.warmup_seconds:
                if self.config.warmup_queries:
                    _run_queries(n_queries=self.config.warmup_queries)
                else:
                    _run_queries(for_seconds=self.config.warmup_seconds)

            # aggregate latencies
            latency_summary = {
                "count": len(per_query_latencies),
                "p50_ms": statistics.median(per_query_latencies) if per_query_latencies else None,
                "p95_ms": (sorted(per_query_latencies)[int(0.95 * len(per_query_latencies))] if per_query_latencies else None),
                "p99_ms": (sorted(per_query_latencies)[int(0.99 * len(per_query_latencies))] if per_query_latencies else None),
                "mean_ms": statistics.mean(per_query_latencies) if per_query_latencies else None,
            }

            # measurement phase: if configured, run either a fixed number of queries or for a duration and compute QPS
            measurement_count = 0
            measurement_duration = None
            if self.config.measurement_queries or self.config.measurement_seconds:
                m_start = time.perf_counter()
                if self.config.measurement_queries:
                    # run measurement_queries queries and record latencies
                    for idx in range(self.config.measurement_queries):
                        q = self.queries[idx % len(self.queries)]
                        qtext = q.get("query")
                        qv_local = ep.embed(qtext)
                        s = time.perf_counter()
                        if method == "exact":
                            _ = exact.search(qv_local, top_k=max(self.config.top_k))
                        elif method == "hnsw":
                            ef_search = sweep_params.get("ef_search", self.config.hnsw.ef_search)
                            _ = method_retriever.search(qv_local, top_k=max(self.config.top_k), ef_search=ef_search)
                        elif method == "bm25":
                            _ = method_retriever.search(qtext, top_k=max(self.config.top_k))
                        elif method == "hybrid":
                            _ = method_retriever.search(qtext, qv_local, top_k=max(self.config.top_k))
                        e = time.perf_counter()
                        per_query_latencies.append((e - s) * 1000.0)
                        measurement_count += 1
                else:
                    # run for measurement_seconds
                    end_time = time.perf_counter() + self.config.measurement_seconds
                    i = 0
                    while time.perf_counter() < end_time:
                        q = self.queries[i % len(self.queries)]
                        qtext = q.get("query")
                        qv_local = ep.embed(qtext)
                        s = time.perf_counter()
                        if method == "exact":
                            _ = exact.search(qv_local, top_k=max(self.config.top_k))
                        elif method == "hnsw":
                            ef_search = sweep_params.get("ef_search", self.config.hnsw.ef_search)
                            _ = method_retriever.search(qv_local, top_k=max(self.config.top_k), ef_search=ef_search)
                        elif method == "bm25":
                            _ = method_retriever.search(qtext, top_k=max(self.config.top_k))
                        elif method == "hybrid":
                            _ = method_retriever.search(qtext, qv_local, top_k=max(self.config.top_k))
                        e = time.perf_counter()
                        per_query_latencies.append((e - s) * 1000.0)
                        measurement_count += 1
                        i += 1
                measurement_duration = time.perf_counter() - m_start
                # recompute latency summary with measurement latencies
                if per_query_latencies:
                    latency_summary = {
                        "count": len(per_query_latencies),
                        "p50_ms": statistics.median(per_query_latencies) if per_query_latencies else None,
                        "p95_ms": (sorted(per_query_latencies)[int(0.95 * len(per_query_latencies))] if per_query_latencies else None),
                        "p99_ms": (sorted(per_query_latencies)[int(0.99 * len(per_query_latencies))] if per_query_latencies else None),
                        "mean_ms": statistics.mean(per_query_latencies) if per_query_latencies else None,
                    }
            # compute qps
            qps = None
            if measurement_duration and measurement_duration > 0:
                qps = measurement_count / measurement_duration

            # resource snapshot at end
            resources = self.monitor.snapshot()

            entry = {"method": method, "latency_summary": latency_summary, "resources": resources, "queries": method_results, "sweep_params": sweep_params}
            if measurement_duration is not None:
                entry["measurement_duration_s"] = measurement_duration
                entry["measurement_qps"] = qps
                entry["measurement_count"] = measurement_count
            results["experiments"].append(entry)

        exp_id = str(uuid.uuid4())
        self.writer.write(exp_id, results)
        return {"experiment_id": exp_id, "results": results}
