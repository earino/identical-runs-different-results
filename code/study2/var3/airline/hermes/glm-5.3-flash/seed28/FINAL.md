# FINAL — airline delay AUC (xgboost-autoresearch, scenario 2)

Best Eval AUC: **0.7204** (commit `1481f35`, validated: `CONTRACT OK`, predict_proba AUC 0.7204).
Baseline was 0.7141 → +0.0063. 40/40 experiments used, ~3,050 of 18,000 CPU-seconds spent.

## Final architecture

Ensemble of 20 XGBoost classifiers (30 trees, depth ∈ {4,6,8}, lr ∈ {0.08,0.1,0.12},
subsample ∈ {0.6,0.7,0.8}, colsample_bytree ∈ {0.6,0.7,0.8}, colsample_bynode 0.7,
hist, enable_categorical) trained on the full 2005 train set, fused with weights derived
from 5-fold out-of-fold AUC per member: `w ∝ 0.35 + 0.65·clip(oof_auc−0.65)⁸` (every
member keeps ≥35% weight). `predict_proba(df)` rebuilds all features inside `prepare()`.

## Changes that mattered most

1. **Small-tree bagging with hyperparameter jitter** (exp 10–20): 5→20 members with
   depth/subsample/lr jitter, +0.0048 over baseline. Small trees (30) generalize across
   the 2005→2006 shift; diversity, not depth, buys AUC.
2. **colsample_bynode=0.7** (exp 17): per-split feature sampling decorrelates members, +0.0008.
3. **OOF-weighted member fusion** (exp 32/34): down-weight structurally weak members
   using target-free out-of-fold AUC, +0.0005 over the plain mean.
4. **dep_hour / dep_tod / night / log_dist** (exp 11): scheduled departure hour is the
   dominant delay driver; encoding it as a number next to the native categoricals, +0.0008.
5. **carrier×hour crossed categorical** (exp 12): per-carrier time-of-day delay patterns, +0.0005.

## What did not help

- **Target encoding** (exp 5/6, smoothed, train-only): 0.7022 — leak on own rows and
  stale 2005 statistics for 2006.
- **Route (Origin_Dest) categorical** (exp 15) and **larger crosses** (org_month, dow_hour,
  exp 19): 0.7106/0.7120 — too many levels, overfit year-specific airport pairs.
- **Deeper/longer single models with early stopping** (exp 2/3): 0.7097/0.7110 — ES picks
  the year-overfit point; time-split-free eval rewards small trees.
- Also flat: numeric month/day cyclical features (27), rank-average fusion (23),
  max_delta_step (25), bag of 40 (31), two-view feature partitions (30), aggregate
  frequency/mean-distance features (22).

## With more budget

Single-member capacity kept paying to the timeout wall (60→80→100 trees: 0.7200→0.7203→0.7204,
and 130 trees was timeout-neutral at 4-fold OOF but scored equal). I would raise members to
100 trees × 30 with a 2-fold OOF pass, and add leak-free route-level context features
(route traffic percentile instead of route ID; dest × time-band pair) — the crossed-categorical
direction failed only at high cardinality.
