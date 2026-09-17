# FINAL — airline XGBoost (autoresearch harness, scenario 2)

**Best Eval AUC: 0.7248** (baseline 0.7141, +0.0107). Best commit `cf733d4`, validated: `predict_proba` on
`data/eval.csv` with the target removed reproduces 0.7248 → `CONTRACT OK`.

Model: ensemble of three XGBoost classifiers (max_depth 3/4/5, 250 trees, lr 0.05, hist, native categorical
handling), predictions averaged. All feature engineering is inside `prepare(df)` and every statistic is fit on
`data/train.csv` only, so the hidden holdout is transformed identically.

## Changes that mattered most
1. **Ordinal calendar parsing + DepTime decomposition.** Parsing `c-N` strings to ints and replacing raw
   `DepTime` with `dep_hour`, `dep_min`, `dep_minutes` (~0.718 → 0.7191). Feature importance is dominated by
   `dep_minutes` (0.42) and `dep_hour` (0.22).
2. **Seasonality/weekend features** `day_of_year`, `is_weekend` (→ 0.7199), replacing the now-redundant
   `Month`/`DayofMonth`/`DayOfWeek` columns (they received zero splits).
3. **Shallow, regularized capacity + depth ensemble.** The 2005→2006 shift punishes capacity: 500 deep trees
   scored 0.7098, depth 3 alone 0.7119, while depth 4 with ~150 low-lr trees scored 0.7171. Averaging depth
   3/4/5 models added a small but consistent gain.
4. **Airport congestion features (the biggest late gain).** `origin/dest_hour_cnt` counts and their
   size-normalized **hour fractions** (`cnt / airport_total`) → 0.7203/0.7225; **±1h windowed congestion**
   and **window fractions** → 0.7239/0.7248. These capture "how busy is this airport at this time of day,
   relative to its own traffic" and transfer across years far better than airport identity.
5. **Re-tuning capacity after feature growth:** raising the ensemble from 150 to 250 trees gave 0.7231 once
   the congestion features were present.

## Things that did not help (reverted)
- **Route as a raw high-cardinality categorical** — catastrophic (0.7115), and route/carrier-hour counts and
  airport-day counts also hurt (0.7187, 0.7115). Identity features overfit the year boundary.
- **Smoothed target encoding** (OOF, k=50) of carrier/origin/dest/route: neutral (0.7196) and added complexity.
- **Extra capacity / regularization knobs**: 400+ trees, 500-tree lr0.03, subsample/colsample, min_child_weight,
  reg_lambda, DART, one-hot carrier, per-depth tree counts, cyclic sin/cos encodings — all neutral or worse.

## What I would try with more budget
The signal is overwhelmingly time-of-day × airport-traffic, and the winning features were congestion *shapes*
(relative shares), so I would push further in that direction: congestion measured on a finer time grid
(15/30-minute bins and a smooth kernel over scheduled departure time rather than integer-hour buckets), the
same congestion evaluated at the implied scheduled **arrival** time (dep time + distance-based flight time,
since arrival congestion backs up to departures), and a weather-proxy-style "traffic anomaly" feature (today's
count at an airport-hour vs its historical average). On the modeling side I would try stacking the depth
3/4/5 ensemble with a logistic meta-learner trained on out-of-fold predictions, and calibrate model weights on
an internal time split rather than uniform averaging. I would avoid any further high-cardinality identity
features or deep trees, since every experiment in that direction lost to the 2005→2006 shift.
