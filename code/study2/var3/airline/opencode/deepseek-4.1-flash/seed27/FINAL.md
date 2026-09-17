# FINAL — airline XGBoost autoresearch

**Best Eval AUC: 0.7438** (commit `2a37e00`, experiment #39)
Baseline: 0.7141 (experiment #1). Total gain: **+0.0297 AUC** over 40 experiments.

## Final model

Three-seed ensemble of `xgboost.XGBClassifier` (hist, native categoricals), averaged
probabilities. Per model: `grow_policy="lossguide"`, `max_leaves=1023`, `n_estimators=400`,
`learning_rate=0.025`, `min_child_weight=5`, `subsample=0.8`, `colsample_bytree=0.4`,
`reg_lambda=1.0`. All feature engineering lives in `prepare()`, so `predict_proba` reproduces it
on unseen rows; categorical levels and frequency maps are fit on `data/train.csv` only.

## Changes that mattered most

1. **Feature engineering in `prepare()`:** decomposed `DepTime` (raw `hhmm` integer is not ordinal
   across hour boundaries) into hour / minute / minutes-of-day, parsed the `c-<n>` string columns to
   integers, and added train-fitted frequency encodings for Origin, Dest, UniqueCarrier and
   Origin_Dest route. +0.0032 at fixed capacity.
2. **`grow_policy="lossguide"` with a large leaf budget:** lossguide trees were strictly better than
   depthwise (`max_depth=4`). Going 31 → 63 → 127 → 255 → 511 → 1023 leaves monotonically improved
   eval AUC up to ~1023 (0.7224 → 0.7420 single model). This was the largest single lever.
3. **Aggressive feature subsampling (`colsample_bytree=0.4`):** the biggest hyperparameter jump,
   +0.0065 single-model over the 0.8 default, and robust across the sweep (0.8 < 0.6 < 0.4 > 0.25).
   With only ~14 features, 40% per tree decorrelates splits and strongly regularizes.
4. **Lower learning rate + more trees:** `lr=0.1, n=100` → `lr=0.025, n=400`, a small but consistent
   gain at high leaf budget.
5. **Seed ensembling (3 models):** +0.0018 over the best single model, and should transfer at least as
   well to the hidden holdout because it only reduces variance.

## What did NOT help

1. **More trees at low regularization / depthwise depth 6:** 500 trees at `lr=0.05` scored *below* the
   30-tree baseline (0.7119 vs 0.7141) — the 2005→2006 time shift punishes high-capacity, unregularized
   fits.
2. **High-cardinality `Route` as a native categorical:** collapsed to 0.7036 (severe overfitting);
   route was only useful as a frequency encoding.
3. **Smoothed target encoding** of Origin/Dest/Carrier (0.7201, below the 0.7211 then-best) and
   calendar/cyclical features (day-of-year, is_weekend, sin/cos: 0.7198). Neither transferred.
4. Hyperparameter micro-tuning that was neutral-to-negative: `min_child_weight` 30 (0.7195), depth 3
   (0.7184), `reg_lambda=5` (0.7303 ≈ baseline), `subsample=0.6` (0.7376).

## Overfitting / robustness notes

Eval is 2006 slice 1 and the hidden holdout is later 2006, so eval is a fair proxy — but early results
showed the model overfits the 2005→2006 distribution shift, which is why every gain came from
*regularization* rather than raw capacity. The final config (lossguide + 40% column sampling + 3-seed
average) trades away single-model memorization for a smoother, more transferable fit.

## With more budget I would try

A wider seed ensemble (8–12 models) and diversity across leaf budgets (511/1023/1536) averaged
together; a time-aware validation split (train on early 2005, validate on late 2005) to select
hyperparameters without touching eval at all; and interaction features between carrier and
time-of-day scheduled banks, which the current leaf-limited trees can only partially represent.
