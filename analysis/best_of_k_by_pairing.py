#!/usr/bin/env python3
"""Best-of-k for each Study 2 pairing separately (the paper's table pools the six).

Same policy and exact computation as best_of_k.py, within one pairing: k attempts from its observed runs with
replacement, noncompliant attempts rejected, the one whose delivered code scores best on the evaluation set kept, and
its holdout AUC reported. "Optimism" is how much higher the mean kept score would be if the winner were chosen on the
holdout itself, the largest over k.

Usage (repo root): python experiments/run_variance/best_of_k_by_pairing.py [CELLS_CSV [FINAL_EVAL_CSV]]
  FINAL_EVAL_CSV comes from export_final_eval.py (default results/variance/final_eval.csv)
"""
import statistics as st, sys

from best_of_k import KS, load, mean, pooled, quantile

CELLS = sys.argv[1] if len(sys.argv) > 1 else "results/variance/cells.csv"
FINAL = sys.argv[2] if len(sys.argv) > 2 else "results/variance/final_eval.csv"
NAME = {"pi": "pi", "hermes": "Hermes", "opencode": "OpenCode",
        "glm-5.3-flash": "GLM-5.3 Flash", "deepseek-4.1-flash": "DeepSeek 4.1 Flash"}

pairs = load(CELLS, FINAL)
print("best-of-k within each pairing, exact; median kept holdout AUC, chosen on the evaluation score\n")
print(f"{'pairing':30s} {'1':>7s} {'3':>7s} {'5':>7s} {'10':>7s}   gain 1->3  gain 1->10   max optimism   >=1 compliant at k=1")
for key in sorted(pairs, key=lambda k: st.mean(h for h, _, ok in pairs[k] if ok)):
    one = {key: pairs[key]}
    med = {k: round(quantile(pooled(one, k)[1], .5), 4) for k in KS}
    opt = max(mean(pooled(one, k, "holdout")[1]) - mean(pooled(one, k)[1]) for k in KS)
    print(f"{NAME[key[0]] + ' / ' + NAME[key[1]]:30s} " + " ".join(f"{med[k]:7.4f}" for k in KS)
          + f"   {med[3] - med[1]:+.4f}    {med[10] - med[1]:+.4f}      {opt:+.5f}      {pooled(one, 1)[0]:.1%}")
