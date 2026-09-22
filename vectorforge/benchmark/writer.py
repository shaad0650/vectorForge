from __future__ import annotations
from typing import Any, Dict
import json
from pathlib import Path
import csv


class ResultWriter:
    def __init__(self, out_dir: str = "results"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    def write(self, experiment_id: str, results: Dict[str, Any]):
        p = self.out_dir / f"{experiment_id}.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        # write a CSV summary with one row per method/experiment and aggregated stats
        csvp = self.out_dir / f"{experiment_id}_summary.csv"
        rows = []
        for exp in results.get("experiments", []):
            if exp.get("skipped"):
                rows.append({
                    "method": exp.get("method"),
                    "skipped": True,
                    "reason": exp.get("reason"),
                })
                continue
            method = exp.get("method")
            lat = exp.get("latency_summary", {})
            resources = exp.get("resources", {})
            # allow sweep parameters to be present
            row = {
                "method": method,
                "count": lat.get("count"),
                "mean_ms": lat.get("mean_ms"),
                "p50_ms": lat.get("p50_ms"),
                "p95_ms": lat.get("p95_ms"),
                "p99_ms": lat.get("p99_ms"),
                "rss": resources.get("rss"),
            }
            # add any sweep fields
            for k, v in exp.get("sweep_params", {}).items():
                row[k] = v
            rows.append(row)
        if rows:
            # build a union of all keys for csv header ordering
            fieldnames = []
            for r in rows:
                for k in r.keys():
                    if k not in fieldnames:
                        fieldnames.append(k)
            with open(csvp, "w", newline="", encoding="utf-8") as cf:
                writer = csv.DictWriter(cf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
