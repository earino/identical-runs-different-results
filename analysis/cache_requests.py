#!/usr/bin/env python3
"""Section 4.9: where Claude Code's prompt cache failed, request by request (Study 1, Ollama Cloud rows).

Reads data/study1/claude_requests.csv (one row per request, token counts only, exported from the session transcripts
by claude_requests_export.py) and prints, per run: the first request's prompt size, the largest number of tokens any
request read back from the cache, the largest prompt, and how many requests read back more than half their prompt.
On DeepSeek V4 Flash no request reads back much; on V4.1 Flash the same agent reads back most of its prompt from the
second request on, at far larger prompt sizes, so context size does not explain the failure.

Usage (repository root): python analysis/cache_requests.py
"""
import csv
from collections import defaultdict

runs = defaultdict(list)
for r in csv.DictReader(open("data/study1/claude_requests.csv")):
    if r["provider"] == "ollama":
        runs[(r["model"], r["seed"])].append(r)
print(f"{'model':24} {'seed':>4} {'requests':>8} {'first prompt':>12} {'max cache read':>14} {'max prompt':>10} {'>half cached':>12}")
totals = {}
for (model, seed), v in sorted(runs.items()):
    prompt = [int(r["prompt_tokens"]) for r in v]; read = [int(r["cache_read_tokens"]) for r in v]
    half = sum(1 for p, c in zip(prompt, read) if p and c / p > 0.5)
    print(f"{model:24} {seed:>4} {len(v):>8} {prompt[0]:>12} {max(read):>14} {max(prompt):>10} {half:>7} of {len(v)}")
    tot = totals.setdefault(model, [0, 0, 0, 0])
    tot[0] += half; tot[1] += len(v); tot[2] = max(tot[2], max(read)); tot[3] = max(tot[3], max(prompt))
print()
for model, (half, n, mr, mp) in totals.items():
    print(f"{model}: {half} of {n} requests read back more than half their prompt; largest cache read {mr}, largest prompt {mp}")
