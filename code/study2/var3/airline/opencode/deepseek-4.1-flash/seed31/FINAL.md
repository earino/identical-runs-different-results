# FINAL — airline XGBoost (autoresearch)

**Best Eval AUC: 0.7377** (baseline 0.7141; +0.0236).
Final model: 5-model XGBoost ensemble (`hist`, native categoricals), depths 12/16/20 at lr 0.05,
300 trees, plus two subsampled/colsampled depth-16/20 members. Contract validated (`CONTRACT OK`).

## Changes that mattered most

1. **Dropping `Month` and `DayofMonth` categoricals.** The single biggest win (+0.0039 over baseline
   alone). Categorical month/day let the model memorise 2005-specific weather/seasonal noise that does
   not transfer to the 2006 eval. Day-of-week *does* transfer, so it was kept.
2. **Much deeper trees.** With the overfitting columns removed, accuracy rose monotonically with depth:
   6 -> 0.7141, 10 -> 0.7254, 12 -> 0.7281, 16 -> 0.7322 (peak), 24 -> 0.7316. Depth 16 captured
   DepTime x carrier x airport interactions that shallow trees could not.
3. **More trees at a lower learning rate.** 300 trees @ lr 0.05 (0.7346) beat 120 @ 0.1 (0.7194) and
   600 @ 0.03 (0.7340). The current best single-model setting.
4. **Multi-depth bagging.** Averaging depths 12/16/20 lifted a single depth-16 model from 0.7346 to
   0.7362, and adding subsampled/colsampled members pushed it to 0.7377 — variance reduction that
   matters for the 1M-row hidden holdout.
5. **Keeping native categorical handling** for `UniqueCarrier`/`Origin`/`Dest` (one-hot was worse:
   0.7102 vs 0.7141 at baseline depth).

## Things that did NOT help

- **Extra engineered features**: DepTime hour/minute/cyclic, route as a categorical, frequency
  encodings, and cyclic Month/day-of-month all failed to beat the raw columns; high-cardinality
  route categorical (4198 levels) and any reintroduction of month seasonality hurt substantially.
- **Regularisation / subsampling as a single model**: `min_child_weight`, `gamma`, `reg_lambda`,
  `subsample`, `colsample_bytree`, and DART all lowered AUC when applied to the standalone model.
- **Shallow/one-hot/seed variants**: depth 4 (0.6998), depth 7 on the *uncleaned* features (0.6989),
  one-hot encoding (0.7102), and seed changes (identical 0.7141) confirmed depth-6-on-raw was limited.

## What I would try with more budget

Now that depth and feature cleaning are understood, the next axis is **out-of-fold target encoding**
for `Origin`, `Dest`, `UniqueCarrier` and `Origin_Dest` (fit strictly on training folds, full-train
statistics at inference), which may replace some of the deep-tree partitioning with smoother,
year-robust priors. I would also try `grow_policy="lossguide"` with a large `max_leaves` and tuned
`max_cat_threshold` to control high-cardinality splits, a temporal validation split inside 2005 for
early stopping instead of a fixed tree count, and a larger/deeper bagged ensemble (8-10 members with
varied seeds and feature fractions) trained within the 120 s limit. Finally, per-carrier or
per-origin residual models could capture the strong carrier-level delay propensity (0.15-0.64 target
rate) more directly.
