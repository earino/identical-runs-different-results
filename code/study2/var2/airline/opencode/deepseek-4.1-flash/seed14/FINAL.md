# FINAL — XGBoost airline delay classifier

**Best Eval AUC: 0.7454** (commit `080c322`, `subsample=0.5 colsample_bytree=0.5`, 50-member depth-diverse bag).
Baseline was 0.7141, so the loop added **+0.0313** over 40 experiments. Contract validated on the
best commit (`validate.sh` → `CONTRACT OK`, eval AUC 0.7454 reproduced through `predict_proba`).

## Changes that mattered most

1. **Temporal feature engineering (+0.0037).** Parsed `DepTime` (hhmm) into hour / minute / minutes-of-day
   plus sin/cos terms, and decoded the `c-<n>` date parts into ordered numerics with cyclical encodings.
   Replacing the raw integer + unordered categorical date parts was the first real win.
2. **Smoothed out-of-fold target encoding of interaction keys (+0.0081).** `(Origin, hour)`,
   `Route (Origin_Dest)`, and `(Route, hour)`, fit on train only, 5-fold OOF for the training matrix and
   the full-train map at inference. This single change gave the biggest jump (0.7253 → 0.7334).
3. **Congestion count features (+0.003).** Flight counts per `(airport, hour)`, `(route)`, `(route, hour)`
   from train — a traffic-volume proxy that transfers across years.
4. **Depth-diverse bagging (+~0.003).** 40–60 `XGBClassifier` members at depths `[16..20]`, averaged.
   Diversity mattered more than any single model; deeper trees only paid off once the TE features existed.
5. **Strong stochastic subsampling (+0.004).** `subsample=0.5, colsample_bytree=0.5` improved monotonically
   from 0.8 → 0.5 (0.7404 → 0.7432 → 0.7453 → 0.7454).

## Things that did not help

- **More capacity / no regularization:** 400 trees, 60 trees @ lr 0.05, and early stopping on an internal
  split all underperformed 30 trees @ lr 0.1; removing `min_child_weight` blew the 120 s cap.
- **Raw high-cardinality categoricals:** `Route` as a native categorical (0.7093) and re-adding the date
  parts as categoricals (0.7180) both regressed — categorical splits do not transfer across the year shift.
- **Global / coarse interaction encodings:** global `o/d/c` TE, 2h-bucket TEs, and week-interaction TEs
  added ≈0.000–0.001 and were discarded.

## What I would try with more budget

Tune the TE smoothing jointly with depth (alpha was only probed coarsely), give each bag member a
different OOF fold assignment for extra TE diversity, try `grow_policy="lossguide"` with a leaf cap, and
bag many more members now that the winning `subsample=0.5` config trains in ~59 s (the 120 s cap, not the
40-experiment cap, was the binding constraint at the end). A count-weighted or leave-one-out target
encoder might also reduce the OOF/full-map mismatch the deep trees exploit.
