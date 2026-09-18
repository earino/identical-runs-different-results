#!/usr/bin/env python3
"""Best-of-k: what attempting a job k times and keeping the best compliant result buys (the paper's Study 2 table).

Each draw picks one of the six pairings at random, then k attempts from its observed runs, with replacement.
Noncompliant attempts are rejected first. Among the rest the winner is chosen two ways, on the same draws:

  holdout   chosen on holdout AUC and reported on holdout AUC (the published table);
  eval      chosen on the evaluation-set AUC of the code the run delivered (its final train.py, which in 18 of the
            312 runs was not its best experiment; export_final_eval.py extracts it), and reported on holdout AUC.

The second is the out-of-sample version: the set used to pick the winner is not the set that scores it. The two
agree to four decimals because a run's score on the 1M-row holdout has a standard error of about 0.0005, twenty
times smaller than the spread between runs, so choosing on it adds almost no winner's-curse optimism; the script
prints the measured gap so it can be checked rather than argued.

Usage (repo root): python experiments/run_variance/best_of_k.py [CELLS_CSV [FINAL_EVAL_CSV]]
  FINAL_EVAL_CSV comes from export_final_eval.py (default results/variance/final_eval.csv)
"""
import csv, random, statistics as st, sys
from collections import defaultdict

import numpy as np

CELLS = sys.argv[1] if len(sys.argv) > 1 else "results/variance/cells.csv"
FINAL = sys.argv[2] if len(sys.argv) > 2 else "results/variance/final_eval.csv"
final_eval = {(r["harness"], r["model"], r["seed"]): float(r["final_eval_auc"]) for r in csv.DictReader(open(FINAL))}
DRAWS, SEED, KS = 20000, 0, (1, 3, 5, 10)

pairs = defaultdict(list)
for r in csv.DictReader(open(CELLS)):
    if r["counted"] != "True" or r["status"] != "scored" or not r["holdout_auc"]:
        continue
    compliant = r["refit_suspect"] != "True" and r["frame_stats"] != "True"
    pairs[(r["harness"], r["model"])].append((float(r["holdout_auc"]), final_eval[(r["harness"], r["model"], r["seed"])],
                                              compliant))
keys = sorted(pairs)

rng = random.Random(SEED)
res = {}
for k in KS:
    by_holdout, by_eval, same, drawn = [], [], 0, 0
    for _ in range(DRAWS):
        attempts = [a for a in rng.choices(pairs[rng.choice(keys)], k=k) if a[2]]   # reject noncompliant first
        drawn += 1
        if not attempts:
            continue
        h, e = max(attempts, key=lambda a: a[0]), max(attempts, key=lambda a: a[1])
        by_holdout.append(h[0]); by_eval.append(e[0]); same += h == e
    res[k] = (len(by_holdout) / drawn, np.quantile(by_holdout, [.5, .05, .95]), np.quantile(by_eval, [.5, .05, .95]),
              st.mean(by_holdout) - st.mean(by_eval), same / len(by_holdout))

print(f"best-of-k over compliant runs, {len(keys)} pairings, {DRAWS:,} draws per k (seed {SEED})\n")
print(f"{'k':>3}  {'>=1 compliant':>13}   {'chosen on holdout: median  p5      p95':38}   {'chosen on eval: median  p5      p95':36}"
      f"   optimism   same run")
for k in KS:
    ok, qh, qe, gap, same = res[k]
    print(f"{k:>3}  {ok:>13.1%}   {'':19}{qh[0]:.4f}  {qh[1]:.4f}  {qh[2]:.4f}   {'':16}{qe[0]:.4f}  {qe[1]:.4f}  {qe[2]:.4f}"
          f"   {gap:+.5f}   {same:.0%}")
q = {k: np.round(res[k][1], 4) for k in KS}   # differences of the table's own rounded values, so text and table agree
print(f"\nmedian kept: three attempts {q[3][0] - q[1][0]:+.4f} over one, ten attempts {q[10][0] - q[1][0]:+.4f}")
print(f"one attempt to ten: 5th percentile {q[10][1] - q[1][1]:+.4f}, 95th percentile {q[10][2] - q[1][2]:+.4f}")
print(f"optimism from choosing on the holdout (mean kept, holdout-chosen minus eval-chosen): "
      f"at most {max(res[k][3] for k in KS):+.5f} AUC")
