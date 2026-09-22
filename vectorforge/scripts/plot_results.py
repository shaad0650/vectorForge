"""Plot consolidated results into PNGs: recall vs latency and recall vs memory.
Requires matplotlib and pandas (install via `pip install pandas matplotlib`).
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
CONSOLIDATED = RESULTS / "consolidated_results.csv"

def load():
    return pd.read_csv(CONSOLIDATED)

def plot_recall_vs_p95(df: pd.DataFrame):
    # pick recall@10 vs p95_ms for methods hnsw/exact/bm25/hybrid
    df2 = df[df['method'].isin(['hnsw','exact','bm25','hybrid'])]
    df2 = df2.copy()
    df2['recall@10'] = pd.to_numeric(df2['recall@10'], errors='coerce')
    df2['p95_ms'] = pd.to_numeric(df2['p95_ms'], errors='coerce')
    fig, ax = plt.subplots()
    for name, g in df2.groupby('method'):
        ax.scatter(g['p95_ms'], g['recall@10'], label=name)
    ax.set_xlabel('p95 latency (ms)')
    ax.set_ylabel('Recall@10')
    ax.legend()
    out = RESULTS / 'recall10_vs_p95.png'
    fig.savefig(out)
    print('wrote', out)

def plot_rss_vs_recall(df: pd.DataFrame):
    df2 = df[df['method'].isin(['hnsw','exact','bm25','hybrid'])]
    df2 = df2.copy()
    df2['recall@10'] = pd.to_numeric(df2['recall@10'], errors='coerce')
    df2['rss'] = pd.to_numeric(df2['rss'], errors='coerce')
    fig, ax = plt.subplots()
    for name, g in df2.groupby('method'):
        ax.scatter(g['rss']/1024/1024, g['recall@10'], label=name)
    ax.set_xlabel('RSS (MB)')
    ax.set_ylabel('Recall@10')
    ax.legend()
    out = RESULTS / 'recall10_vs_rss.png'
    fig.savefig(out)
    print('wrote', out)

def main():
    if not CONSOLIDATED.exists():
        print('run aggregate_results.py first')
        return
    df = load()
    plot_recall_vs_p95(df)
    plot_rss_vs_recall(df)

if __name__ == '__main__':
    main()
