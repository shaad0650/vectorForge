"""Simple CLI to run a benchmark using vectorforge.benchmark"""
from __future__ import annotations
import argparse
from vectorforge.benchmark.config import BenchmarkConfig
from vectorforge.benchmark.runner import BenchmarkRunner


def main():
    p = argparse.ArgumentParser()
    p.add_argument("config", help="path to benchmark yaml config")
    args = p.parse_args()
    cfg = BenchmarkConfig.load(args.config)
    runner = BenchmarkRunner(cfg)
    out = runner.run()
    print("Done.", out.get("experiment_id"))


if __name__ == "__main__":
    main()
