from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import yaml


@dataclass
class HNSWConfig:
    M: int = 16
    ef_construction: int = 200
    ef_search: int = 50


@dataclass
class BenchmarkConfig:
    name: str = "default"
    dataset_path: Optional[str] = None  # path to npz with vectors/ids/texts
    queries_path: Optional[str] = None  # JSONL with {"query":..., "relevant": [ids]}
    methods: List[str] = field(default_factory=lambda: ["exact", "hnsw", "bm25", "hybrid"])
    top_k: List[int] = field(default_factory=lambda: [5, 10])
    hnsw: HNSWConfig = field(default_factory=HNSWConfig)
    hybrid_alpha: List[float] = field(default_factory=lambda: [0.25, 0.5, 0.75])
    queries_limit: Optional[int] = 100
    allow_fallback_embeddings: bool = False
    # warmup/measurement controls (seconds or query counts)
    warmup_seconds: Optional[int] = None
    measurement_seconds: Optional[int] = None
    warmup_queries: Optional[int] = None
    measurement_queries: Optional[int] = None
    # parameter sweep spec
    sweep: Dict[str, Dict[str, List[Any]]] = field(default_factory=dict)

    @staticmethod
    def load(path: str) -> "BenchmarkConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # simple mapping
        h = data.get("hnsw", {})
        cfg = BenchmarkConfig(
            name=data.get("name", "default"),
            dataset_path=data.get("dataset", {}).get("path"),
            queries_path=data.get("queries", {}).get("path"),
            methods=data.get("methods", ["exact"]),
            top_k=data.get("retrieval", {}).get("top_k", [5, 10]),
            hnsw=HNSWConfig(M=h.get("M", 16), ef_construction=h.get("ef_construction", 200), ef_search=h.get("ef_search", 50)),
            hybrid_alpha=data.get("hybrid", {}).get("alpha", [0.5]),
            queries_limit=data.get("queries", {}).get("limit", 100),
            allow_fallback_embeddings=data.get("embeddings", {}).get("allow_fallback", False),
            warmup_seconds=data.get("measurement", {}).get("warmup_seconds"),
            measurement_seconds=data.get("measurement", {}).get("measurement_seconds"),
            warmup_queries=data.get("measurement", {}).get("warmup_queries"),
            measurement_queries=data.get("measurement", {}).get("measurement_queries"),
            sweep=data.get("sweep", {}),
        )
        return cfg
