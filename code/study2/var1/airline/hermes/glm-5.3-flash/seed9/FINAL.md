# Final report — airline delay, XGBoost autoresearch

**Best Eval AUC: 0.7179** (baseline 0.7141, +0.0038). Head commit `51c6a9b`, `CONTRACT OK` via
validate.sh. All 40 experiments used; ~450 CPU-seconds of the 18,000 budget spent.

## What mattered most

1. **Fixed-size model, full training data, no early stopping.** Eval is 2006 data, train is 2005 — a
   distribution shift. AUC on eval *peaks around 50 trees @ lr 0.1, depth 6* and decays with more
   fitting (100 trees 0.7136, 400 trees 0.7021). Early stopping on a random 10% validation split
   further cost ~0.005 vs training on all 100k rows.
2. **Regularization triple: subsample 0.85, colsample_bytree 0.8, reg_lambda 2.0** (+0.010 over plain
   defaults at the same tree count — the largest single-model gain found).
3. **Heterogeneous bagging: 16 XGBoost members with jittered (subsample, colsample, depth) tuples,
   probability-averaged.** This was the main lever after the single-model plateau (+0.004 over the best
   single model: 0.7141 → 0.7179). Diversity from varied members beat simply adding same-config seeds
   (8 homogeneous seeds: 0.7169; 8 heterogeneous members: 0.7177).
4. **Scaling the committee:** 3 → 8 → 12 → 16 members gave 0.7162 → 0.7177 → 0.7178 → 0.7179;
   20/28 members flat-to-worse, so 16 was kept (simplicity criterion).

## What did not help

- **Target encoding** (smoothed, then properly out-of-fold): 0.7053 in-sample, 0.7130 with OOF — both
  below plain native categoricals. XGBoost's `enable_categorical` partitioning already extracts the
  origin/dest/carrier effects; TE mostly added noise under the 2005→2006 shift.
- **Time-cyclical features** (DepMin/Hour/sin/cos, 24h-corrected DepMod variants, OddHour parity
  probe): all within ±0.0003 of the same model without them; raw integer DepTime already provides the
  split points.
- **colsample_bynode=0.5 per member** (0.7168) and **lossguide grow-policy members** (0.7177, equal
  but not better than the simpler all-depthwise ensemble) — both dropped.

## With more budget

Grid-scale the committee members by *drawing* (subsample, colsample, depth, lambda) from a wider range
and growing to 30+ members with per-member bootstrap resampling of rows (proper bagging), plus a
stacked blend (logistic on member outputs via OOF predictions) instead of the plain mean. Also worth
one slot: Origin×Dest route counts and carrier×month frequency features as *categorical* (not TE)
columns, and quantile-bucketed DepTime (15-min bins) as an extra categorical rather than cyclical
numerics.
