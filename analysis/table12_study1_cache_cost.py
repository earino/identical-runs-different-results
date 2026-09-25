#!/usr/bin/env python3
"""Table 12 and the 23-fold cost gap: Study 1, DeepSeek V4 Flash on Ollama Cloud, cached share and rate-card cost.

Per run, from data/study1/cells.csv (the run's own final usage report). A run with no final report (Claude Code seed
3, stalled) takes its figures from data/study1/claude_per_request.csv, reconstructed from the session transcript.
Prints each run, each agent's mean cost, and Claude Code's mean cost over pi's.

Usage (repository root): python analysis/table12_study1_cache_cost.py
"""
import csv, statistics as st
from collections import defaultdict

log = {(r['harness'], r['seed']): r for r in csv.DictReader(open('data/study1/claude_per_request.csv'))
       if r['provider'] == 'ollama' and r['model'].startswith('deepseek-v4-flash')}
runs = defaultdict(list)
for r in csv.DictReader(open('data/study1/cells.csv')):
    if not r['model'].startswith('deepseek-v4-flash'):
        continue
    share, cost, source = r['cached_share'], r['cost_usd_list'], 'final report'
    if not cost and (r['harness'], r['seed']) in log:
        l = log[(r['harness'], r['seed'])]; share, cost, source = l['cached_share'], l['charge_usd'], 'transcript'
    if cost:
        runs[r['harness']].append(float(cost))
        print(f"{r['harness']:9} seed {r['seed']}  cached {float(share):.3f}  cost {float(cost):.4f}  ({source})")
    else:
        print(f"{r['harness']:9} seed {r['seed']}  no record")
print()
means = {h: st.mean(v) for h, v in runs.items()}
for h, m in sorted(means.items(), key=lambda x: x[1]):
    print(f"{h:9} mean cost {m:.4f} over {len(runs[h])} runs")
print(f"\nClaude Code / pi mean cost: {means['claude'] / means['pi']:.1f}")
