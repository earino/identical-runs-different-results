#!/usr/bin/env python3
"""How the Study 2 headline numbers move under each rule for excluding rule-breaking runs.

The paper excludes both kinds of violation: training on the labelled evaluation file (eval_trained) and computing
features from the batch being scored (frame_stats); no run did both. This prints the headline numbers under four
rules (exclude nothing, only the first kind, only the second, both) so a reader can see which conclusions depend on
the exclusion and which do not.

Best-of-k is the paper's policy (best_of_k.py, exact): runs a rule excludes are rejected attempts, and the kept one is
chosen on the evaluation score of its delivered code.

Usage (repo root): python experiments/run_variance/sensitivity_violations.py [CELLS_CSV [FINAL_EVAL_CSV]]
"""
import csv, statistics as st, sys
from collections import defaultdict

from best_of_k import load, pooled, quantile

CELLS = sys.argv[1] if len(sys.argv) > 1 else "results/variance/cells.csv"
FINAL = sys.argv[2] if len(sys.argv) > 2 else "results/variance/final_eval.csv"
rows = [r for r in csv.DictReader(open(CELLS)) if r["counted"] == "True" and r["status"] == "scored" and r["holdout_auc"]]
RULES = {"exclude nothing": lambda r: True,
         "exclude eval-file training only": lambda r: r["eval_trained"] != "True",
         "exclude batch features only": lambda r: r["frame_stats"] != "True",
         "exclude both (the paper)": lambda r: r["compliant"] == "True"}

print(f"{'rule':34s} {'runs':>5s} {'span of means':>14s} {'median SD':>10s} {'best':>7s} "
      f"{'best-of-3 med':>13s} {'best-of-10 med':>14s}")
for name, keep in RULES.items():
    pairs = defaultdict(list)
    for r in rows:
        if keep(r):
            pairs[(r["harness"], r["model"])].append(float(r["holdout_auc"]))
    means = [st.mean(v) for v in pairs.values()]; sds = [st.stdev(v) for v in pairs.values()]
    attempts = load(CELLS, FINAL, keep)
    med = {k: quantile(pooled(attempts, k)[1], .5) for k in (3, 10)}
    print(f"{name:34s} {sum(len(v) for v in pairs.values()):5d} {max(means) - min(means):14.4f} {st.median(sds):10.4f} "
          f"{max(max(v) for v in pairs.values()):7.4f} {med[3]:13.4f} {med[10]:14.4f}")

top = sorted(rows, key=lambda r: -float(r["holdout_auc"]))[:10]
bad = sum(r["compliant"] != "True" for r in top)
print(f"\nof the 10 highest-scoring runs, {bad} broke a task rule")
