#!/usr/bin/env python3
"""Repeat-and-select policies with their uncertainty, Study 3's yield, and the observed configurations side by side.

  BEST-OF-K        the deployable policy: draw k attempts, reject noncompliant ones, keep the one whose delivered code
                   scores best on the evaluation set, report its holdout AUC. Computed exactly (best_of_k.py). The 95%
                   interval for the gain over one attempt is an outer bootstrap: runs resampled within each pairing,
                   2,000 times, the exact gain recomputed on each. Intervals are conditional on the fixed evaluation
                   and holdout sets.
  STUDY 3 ARMS     each model's yield (compliant runs of all runs, unscorable ones counted as failures), compliant
                   quality, and the same policy at one and three attempts.
  CONFIGURATIONS   tokens per run, list-price cost per run, yield, compliant mean and the best-of-3 policy for each of
                   the nine agent-model configurations of Studies 2 and 3.

The evaluation score used to choose is that of the code each run delivered (export_final_eval.py). Costs use the
OpenRouter list prices of 16 September 2026 quoted in the paper, with no cache discount, and exclude reasoning tokens,
which were recovered for Study 2 only; they are for comparing configurations, not bills.

Usage (repo root): python experiments/run_variance/policy_tables.py [S2_CELLS S2_FINAL_EVAL S3_CELLS S3_FINAL_EVAL]
"""
import csv, statistics as st, sys
from collections import defaultdict

import numpy as np

from best_of_k import load, pooled, quantile

PRICE = {"glm-5.3-flash": (0.10, 0.33), "deepseek-4.1-flash": (0.15, 0.60), "glm-5.3": (1.40, 4.40)}   # $ per M, in/out
NAME = {"pi": "pi", "hermes": "Hermes", "opencode": "OpenCode", "glm-5.3-flash": "GLM-5.3 Flash",
        "deepseek-4.1-flash": "DeepSeek 4.1 Flash", "glm-5.3": "GLM-5.3"}
median = lambda pairs, k: quantile(pooled(pairs, k)[1], .5)

PATHS = sys.argv[1:5] if len(sys.argv) >= 5 else ["results/variance/cells.csv", "results/variance/final_eval.csv",
                                                    "results/variance_glm53/cells.csv", "results/variance_glm53/final_eval.csv"]
s2 = load(PATHS[0], PATHS[1])
s3 = load(PATHS[2], PATHS[3])

print("BEST-OF-K, Study 2 (six pairings weighted equally; chosen on the delivered code's evaluation score; exact)")
point = {k: median(s2, k) for k in (1, 3, 5, 10)}
for k in (1, 3, 5, 10):
    print(f"  k={k:2d}  median kept {point[k]:.4f}   at least one compliant {pooled(s2, k)[0]:.5%}")
rng = np.random.default_rng(0); gains = {3: [], 5: [], 10: []}
for _ in range(2000):
    boot = {key: [runs[i] for i in rng.integers(0, len(runs), len(runs))] for key, runs in s2.items()}
    base = median(boot, 1)
    for k in gains:
        gains[k].append(median(boot, k) - base)
for k, g in gains.items():
    lo, hi = np.percentile(g, [2.5, 97.5])
    print(f"  gain, 1 to {k} attempts: {round(point[k], 4) - round(point[1], 4):+.4f}   95% interval {lo:+.4f} to {hi:+.4f} "
          f"(outer bootstrap, 2,000)")

print("\nSTUDY 3 ARMS (GLM-5.3 Flash runs come from Study 2)")
arms = {"GLM-5.3 Flash": {k: v for k, v in s2.items() if k[1] == "glm-5.3-flash"}, "GLM-5.3": s3}
res = {}
for name, pairs in arms.items():
    runs = [x for v in pairs.values() for x in v]; comp = [h for h, _, ok in runs if ok]
    res[name] = (round(median(pairs, 1), 4), round(median(pairs, 3), 4))
    print(f"  {name:14s} yield {len(comp)} of {len(runs)} ({len(comp) / len(runs):.1%})   compliant mean {st.mean(comp):.4f}   "
          f"one attempt: artifact {pooled(pairs, 1)[0]:.1%}, median {res[name][0]:.4f}   best of three: artifact "
          f"{pooled(pairs, 3)[0]:.2%}, median {res[name][1]:.4f}")
print(f"  the larger model's gain under the three-attempt policy: {res['GLM-5.3'][1] - res['GLM-5.3 Flash'][1]:+.4f}; "
      f"under one attempt: {res['GLM-5.3'][0] - res['GLM-5.3 Flash'][0]:+.4f}")

print("\nCONFIGURATIONS (per run; list price, no cache discount, reasoning excluded)")
print(f"  {'agent / model':30s} {'input tok':>10s} {'output tok':>10s} {'cost':>7s} {'yield':>6s} {'mean':>7s} {'best of 3':>9s}")
tok = defaultdict(list)
for path in (PATHS[0], PATHS[2]):
    for r in csv.DictReader(open(path)):
        if r["counted"] == "True":
            tok[(r["harness"], r["model"])].append((float(r["tokens_in"] or 0), float(r["tokens_out"] or 0)))
allc = dict(s2); allc.update(s3)
for key in sorted(allc, key=lambda k: (k[1] != "glm-5.3-flash", k[1], k[0])):
    v = allc[key]; pin, pout = PRICE[key[1]]
    tin, tout = st.mean(t[0] for t in tok[key]), st.mean(t[1] for t in tok[key])
    comp = [h for h, _, ok in v if ok]
    print(f"  {NAME[key[0]] + ' / ' + NAME[key[1]]:30s} {tin / 1e6:9.2f}M {tout / 1e3:9.0f}k ${(tin * pin + tout * pout) / 1e6:6.2f} "
          f"{len(comp) / len(v):6.1%} {st.mean(comp):7.4f} {median({key: v}, 3):9.4f}")
