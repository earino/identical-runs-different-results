#!/usr/bin/env python3
"""Export each run's evaluation-set AUC of the code it delivered, for choosing best-of-k winners out of sample.

cells.csv carries best_eval_auc, the best evaluation score over a run's experiments. The scored artifact is the
run's final train.py, and in 18 of Study 2's 312 runs that was not its best experiment, so its own evaluation score
(final_eval_auc in eval.json) is the right score to choose it on. This writes one row per run with both.

Usage (repo root): python experiments/run_variance/export_final_eval.py [EVALS_DIR] [OUT_CSV]
  defaults: results/variance/evals  ->  results/variance/final_eval.csv
"""
import csv, glob, json, os, sys

EVALS = sys.argv[1] if len(sys.argv) > 1 else "results/variance/evals"
OUT = sys.argv[2] if len(sys.argv) > 2 else "results/variance/final_eval.csv"

rows = []
for f in sorted(glob.glob(os.path.join(EVALS, "*", "airline", "*", "*", "seed*", "eval.json"))):
    e = json.load(open(f))
    rows.append({"box": f.split(os.sep)[-6], "harness": e["harness"], "model": e["model"], "seed": e["seed"],
                 "final_eval_auc": e.get("final_eval_auc"), "best_eval_auc": e.get("best_eval_auc")})
with open(OUT, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
differ = sum(1 for r in rows if r["final_eval_auc"] is not None and r["best_eval_auc"] is not None
             and abs(r["final_eval_auc"] - r["best_eval_auc"]) > 1e-9)
print(f"wrote {OUT}: {len(rows)} runs; final eval differs from best eval in {differ}")
