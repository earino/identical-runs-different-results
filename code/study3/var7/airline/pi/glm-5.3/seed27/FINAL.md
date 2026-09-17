# Final Report — Airline Delay AUC Challenge

**Best Eval AUC: 0.7637** (commit `6ec5fdc`, experiment 17/17 logged runs; baseline 0.7141 → **+0.0496**)

## What mattered most (in order of impact)

1. **Target-encoded categorical keys with empirical-Bayes smoothing** (`(sum + m*prior)/(n + m)`),
   built on train only, unseen keys → prior. Turned the raw-schema XGB into a statistical model of
   operational delay rates. 0.7141 → 0.7225 (#5–#6).
2. **Route × scheduled-time interactions** (`route_hour`, `route_min30`, `route_min15`) plus
   origin/carrier/dest × time keys, each with a `log1p(count)` companion so trees could discount thin
   cells. Single biggest lever: 0.7245 → 0.7519 (#9–#11).
3. **Minute-of-hour (`DepTime % 100`) as a raw numeric** — captures hub departure-bank scheduling;
   plus carrier×origin, carrier×dest, origin×dist-bucket, dist-bucket×hour TEs (distance-bucket edges
   fit on train only). 0.7552 → 0.7617 (#15).
4. **8-member ensemble with per-member dropped feature groups** (each member trains without one TE
   family; predictions averaged). Feature-subset diversity beat seed jitter: 0.7532 → 0.7562 (#12–#14).
5. **2-fold out-of-fold encodings** — building training-matrix TEs from the complementary *half*
   (vs 5/3-fold) while predicting with full-train maps creates a benign shrinkage mismatch that acts
   as regularization: 0.7619 → 0.7637 (#16–#17). This also explained why explicit m-shrinkage sweeps
   were always flat — the OOF/predict mismatch dominated the effective shrinkage.

## What did not help (all reverted or never committed)

- **Residual/stacked TEs** (cell means of model residuals as features): leaked in the first attempt,
  collapsed (iter 0–2) when built properly — both variants strictly worse.
- **Shrinkage-adjacent gadgets**: hierarchical TE shrinkage, per-key m-sweeps, SE-reliability columns,
  kernel-smoothed (±bucket) TEs, quarter-sample maps — all flat or worse.
- **Hyperparameter tuning** (min_child_weight, max_leaves, subsample, colsample, gamma, learning rate,
  early-stopping patience): a flat plateau in every direction once features were fixed; also native
  XGBoost categoricals, monotone constraints, 10-fold OOF, dow/route×calendar keys, and expanding the
  ensemble to 12 members or mixed split-seeds (+0.0003, inside noise).

## With more budget

The eval-vs-holdout gap is now driven by 2005→2006 drift, not capacity: per-segment diagnosis shows
midday hours (base rate ≈ 0.5) and tiny-count routes are intrinsically hard, and month-level AUC varies
with seasonal drift I can't observe on the hidden set. With more budget I would (a) probe drift-robust
encodings — e.g. fit TEs on a rolling last-N-months window of 2005 and check eval AUC by month to
measure how much of the TE signal is seasonal vs structural; (b) test a small two-level stack
(logistic on 8 member OOF probabilities) which needs careful nested OOF to avoid the leak that sank
the residual-TE attempt; and (c) re-run the keep/discard ladder with a 2-fold/seed-average OOF matrix,
since single-split noise (±0.0007) is now the same size as the effects I'm judging.

## Reproducibility

`./validate.sh` → `CONTRACT OK`, eval AUC via `predict_proba` = 0.7637, 88 s training,
~28 s to predict a 1M-row holdout (well inside limits). All encoders fit on `train.csv` only;
`predict_proba` reproduces features from a raw DataFrame.
