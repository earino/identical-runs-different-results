"""Study 1's DeepSeek V4 Pro row: cached share of input per run, and which runs the weekly quota cut off."""
import csv
for r in csv.DictReader(open('data/study1/cells.csv')):
    if r['model'].startswith('deepseek-v4-pro'):
        print(f"{r['harness']:9s} seed {r['seed']}  cached {float(r['cached_share']):.0%}  {'cut off by the quota' if r['provider_error'] else ''}")
