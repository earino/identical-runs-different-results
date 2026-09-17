# Final report — autoresearch XGBoost (airline delay)

**Best Eval AUC: 0.7416** (commit `a2edf9a`), up from the 0.7141 baseline.
Validated: `./validate.sh` → `CONTRACT OK` (eval via `predict_proba` = 0.7416, train 102s).

The final model is an XGBoost-only ensemble of 10 `XGBClassifier`s (probability averaging):
depths 3–6, 800–6400 trees, lr 0.02–0.05, `tree_method=hist`, `max_bin=512`,
`colsample_bylevel=0.3`, `enable_categorical=True`, recency sample weights
`1 + month/12`. All feature engineering and encoder/TE statistics live inside
`prepare()` and are fit on `train` only, so the hidden holdout is handled correctly.

## Changes that mattered most
1. **Categorical treatment of departure time.** Adding departure hour as a categorical
   feature (0.7348 → 0.7375), then a **15-minute time-of-day bin as a categorical**
   (0.7375 → 0.7409) was the single biggest win — letting trees group arbitrary
   (incl. non-adjacent) times instead of only threshold-splitting a numeric clock.
2. **Dropping `Month` and route encoding.** 2005 seasonality and Origin×Dest route
   overfit the time-separated 2006 eval; removing them helped generalization.
3. **Distance binned as categorical** (`Distance // 250`, clipped) added +0.0006.
4. **Heavy regularization + ensembling:** `colsample_bylevel=0.3`, `max_bin=512`, many
   trees, and a 10-member ensemble across capacities (0.7333 → 0.7342 → 0.7348).
5. **Smoothed carrier × time-of-day target encoding** (k=100 smoothing, prior-filled),
   later refined to 30-minute bins (0.7415 → 0.7416).

## Things that did not help (kept/abandoned)
- `Month` as a feature, route/`Origin×Dest` encodings, and Origin/Dest time-bin target
  encodings — all hurt.
- Finer or coarser time bins: 5-min (0.7341) and 30-min (0.7411) both worse than 15-min.
- Classical regularization micro-tuning: `subsample=0.8`, `gamma=0.1`,
  `min_child_weight=2`, `reg_lambda=2`, `colsample_bylevel=0.25` — equal or worse.
- More "diverse" ensembles without extra tree capacity (10 distinct configs: 0.7340).

## With more budget
Try categorical interaction features with cheap target encoding (carrier × tod-bin,
origin × day-of-week) rather than one-hot/categorical to avoid the timeout hit from
high-cardinality categoricals; learn ensemble weights / a small XGBoost stack from
out-of-fold predictions; and pursue proper early stopping via an internal temporal
validation split to set tree counts per member instead of fixed budgets.
