# Final Report — airline (XGBoost, autoresearch harness)

**Best Eval AUC: 0.7505** (baseline 0.7141 → +0.0364). Final `train.py` = commit `870efc4`
(content identical to experiment #29, commit `94a74e4`), verified by `validate.sh` → `CONTRACT OK`
(`predict_proba` reproduces 0.7505 with the target column removed).

## Final model

Ensemble of 16 XGBoost models (averaged predicted probabilities): depth grid {8,10,12,14} ×
learning rate {0.05 (400 trees), 0.1 (200 trees)} × seeds {42,77}, on features
`DayOfWeek? no — DepTime(raw), UniqueCarrier, Origin, Dest, Distance` + `dep_hour`, `dep_min`,
`tod_sin/cos`. Rare carrier/airport levels (<500 train occurrences) grouped into `__OTHER__`
(train-fitted levels; unseen levels map there/NaN). Monotone constraint `+1` on `dep_hour`.
`min_child_weight=2`, `colsample_bytree=0.8`, `tree_method=hist`, native categoricals. All feature
engineering lives in `prepare()`; encoders/stats fit on train only. ~65s train time.

## Changes that mattered most

1. **Dropping Month and DayofMonth** (+0.008 combined): month-of-2005 and day-of-month delay
   patterns do not transfer to 2006 — these 12/31-level categoricals were the largest single
   source of year-shift overfitting.
2. **Grouping rare Origin/Dest/carrier levels** (<500 flights → `__OTHER__`) (+0.015 from 0.7337
   → 0.7489): rare airport/carrier levels memorize training-year noise.
3. **Departure-time decomposition** (`dep_hour` with 2400+ wraparound, `dep_min`, sin/cos)
   (+0.002): robust physical signal; `dep_min` and sin/cos each independently verified.
4. **Depth-diverse ensemble** (d8–14 × 2 lrs × 2 seeds) (+0.025 over single model): deeper members
   are individually worse on eval but average out noise; diversity monotonically helped as the
   feature set got cleaner.
5. **Regularization polish**: `min_child_weight` 10→5→3→2 (+0.0015 total), `colsample_bytree=0.8`,
   monotone `dep_hour` (+0.001).

## Things that did not help

1. **Target encoding** (OOF, smoothed) of carrier/Origin/Dest — and especially of Origin→Dest
   route — always hurt (route TE: −0.01); high-cardinality interaction categoricals
   (carrier×time-bucket, carrier×month) also overfit.
2. **Frequency/count encodings and airport structure features** (hub degree, median distance,
   relative distance): neutral to slightly negative.
3. **Holiday/travel-surge flags, season bucket replacing Month, recency sample weighting toward
   2006, DART booster, logit-averaging, rank-averaging, stacking (OOF meta-XGB)**: all neutral or
   worse; stacking was clearly worse (0.7235 vs 0.7255 mean-blend at the time).

## What I would try with more budget

The dominating discovery was that this 2005→2006 time shift punishes any feature that encodes
calendar or rare-entity detail, so with more budget I would (a) run a proper ablation sweep per
feature × per ensemble member depth (the DayOfWeek-drop interaction with member depth was never
fully mapped — my last five experiments accidentally ran without that drop, so depth grids
(9–15) and `colsample` 0.7 were only measured on the pre-drop base), (b) test per-member feature
views (some members without Distance, some without DayOfWeek) for cheaper diversity than depth
alone, (c) tune the rare-level threshold jointly with `min_child_weight` and tree count via
time-ordered CV inside 2005, and (d) grow a much larger seed pool of the final recipe and select
members by out-of-fold AUC rather than fixed inclusion.
