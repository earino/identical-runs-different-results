# FINAL — autoresearch XGBoost (airline delay, 2005→2006)

## Best result
- **Best Eval AUC: 0.7479** (experiment #18, commit `e0d9ee8`).
- Baseline (30 trees, raw columns): 0.7141. Total gain: **+0.0338 AUC**.
- Final model: probability-average of three XGBoost `hist` models (all native categorical):
  - `A`: depth 12, 700 trees, lr 0.03, colsample_bytree 0.7, λ=50, α=0.5, max_bin 512
  - `A2`: `A` + `colsample_bylevel=colsample_bynode=0.7`
  - `B`: depth 12, 900 trees, lr 0.03, colsample_bytree 0.5, λ=30, α=1.0, γ=0.3, max_bin 256
- Runs in ~80 s (limit 120 s); `predict_proba(df)` reproduces all feature engineering on raw rows.

## Changes that mattered most
1. **Origin × hour-of-day and carrier × hour-of-day categorical interactions.** The single
   biggest lever (+0.015 over the tuned non-interaction model). Airport/carrier-specific
   time-of-day delay propensities generalize across years, unlike airport- or route-level
   *averages*.
2. **Deep trees + heavy L2 regularization.** `max_depth=12` with `reg_lambda` 30–50 was far
   better than depth 4–8 (0.7405 → 0.7456 for a single model). Capacity matters here, and
   regularizing leaf weights rather than limiting depth is what generalizes.
3. **Ensembling diverse deep configs by averaging probabilities** (+0.002). `colsample_bylevel`
   /`colsample_bynode` variants add useful decorrelation.
4. **Dep-time decomposition** (`dep_hour`, `dep_min`, `dep_since_midnight`); `dep_min` turned
   out to be surprisingly important for the deep model (dropping it cost ~0.013).
5. **Native XGBoost categorical handling** with categories fit on train only (unseen → NaN),
   which keeps `predict_proba` self-contained and drift-safe.

## What did NOT help
- **Target/frequency encoding** of Origin/Dest/route/carrier: dropped AUC to ~0.704. The 2005
  target means do not transfer to 2006.
- **Other categorical interactions** (Origin×Month, Carrier×Month, Origin×Dest, Dest×hour,
  weekend-conditioned interactions): all overfit 2005 and reduced eval AUC.
- **Shallow heavily-regularized models, higher learning rates, `subsample<1`,
  `grow_policy=lossguide`, increasing `max_cat_threshold`, minute-as-categorical**: neutral or
  worse.

## With more budget
I would run 5-fold CV bagging over many diverse deep configs (instead of the 2–3 hand-picked
members) to lower ensemble variance without eval-selection bias, and add a stacking layer
(logistic regression on out-of-fold predictions plus a few raw features). I would also sweep
the deep regime on a train-internal temporal validation split rather than eval.csv, to reduce
the risk of overfitting the 100k eval slice, and try pseudo-labeling the large unlabeled pool
if it were available. The feature side looks saturated given only these eight columns; the
remaining headroom is almost entirely model averaging/regularization.
