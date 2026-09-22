"""Aggregate JSON experiment outputs in `results/` into a consolidated CSV.
Produces `results/consolidated_results.csv` with per-experiment per-method metrics.
"""
from __future__ import annotations
import json
from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
OUT = RESULTS / "consolidated_results.csv"

def parse_json(path: Path):
    data = json.load(path.open('r', encoding='utf8'))
    exp_id = path.stem
    rows = []
    for e in data.get('experiments', []):
        method = e.get('method')
        lat = e.get('latency_summary', {})
        res = e.get('resources', {})
        params = {k: e.get(k) for k in ('M','ef_construction','ef_search','alpha')}
        # aggregate per-query metrics (mean recall etc) if present
        qlist = e.get('queries', [])
        mean_recall1 = mean([q.get('metrics',{}).get('recall@1', 0.0) for q in qlist]) if qlist else ''
        mean_recall5 = mean([q.get('metrics',{}).get('recall@5', 0.0) for q in qlist]) if qlist else ''
        mean_recall10 = mean([q.get('metrics',{}).get('recall@10', 0.0) for q in qlist]) if qlist else ''
        mean_mrr = mean([q.get('metrics',{}).get('mrr', 0.0) for q in qlist]) if qlist else ''
        ndcg10 = mean([q.get('metrics',{}).get('ndcg@10', 0.0) for q in qlist]) if qlist else ''

        row = {
            'experiment_id': exp_id,
            'method': method,
            'count': lat.get('count',''),
            'mean_ms': lat.get('mean_ms',''),
            'p50_ms': lat.get('p50_ms',''),
            'p95_ms': lat.get('p95_ms',''),
            'p99_ms': lat.get('p99_ms',''),
            'rss': res.get('rss',''),
            'M': params.get('M',''),
            'ef_construction': params.get('ef_construction',''),
            'ef_search': params.get('ef_search',''),
            'alpha': params.get('alpha',''),
            'recall@1': mean_recall1,
            'recall@5': mean_recall5,
            'recall@10': mean_recall10,
            'mrr': mean_mrr,
            'ndcg@10': ndcg10,
        }
        rows.append(row)
    return rows

def mean(xs):
    xs = [x for x in xs if x is not None and x != '']
    return sum(xs)/len(xs) if xs else ''

def main():
    files = sorted(RESULTS.glob('*.json'))
    allrows = []
    for f in files:
        try:
            rows = parse_json(f)
            allrows.extend(rows)
        except Exception as e:
            print('skip', f, 'err', e)

    if not allrows:
        print('no rows')
        return

    keys = list(allrows[0].keys())
    with OUT.open('w', newline='', encoding='utf8') as fh:
        writer = csv.DictWriter(fh, keys)
        writer.writeheader()
        for r in allrows:
            writer.writerow(r)

    print('wrote', OUT)

if __name__ == '__main__':
    main()
