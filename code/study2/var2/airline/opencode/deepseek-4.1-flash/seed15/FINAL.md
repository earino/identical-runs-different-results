# FINAL — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7279** (experiment #40, commit `90c862f`). Baseline was 0.7141.

`./validate.sh` prints `CONTRACT OK`; `predict_proba(df)` on the raw eval frame
(target removed) reproduces 0.7279, so all feature engineering is inside `prepare()`.

## Changes that mattered most

1. **Time-of-day decomposition + cyclical encodings.** Split `DepTime` (hhmm) into
   `dep_hour`, `dep_min`, `dep_abs = hour*60+min`, and added sin/cos encodings of
   time-of-day (1440 min), month (12), and day-of-week (7). This was the single
   biggest feature-level win (0.7161 -> 0.7189 on top of the dep-time features).
2. **Strong regularization.** `reg_lambda=20`, `reg_alpha=5` on every booster
   (0.7201 -> 0.7241). This dataset has a 2005->2006 time shift, so the model was
   badly overfitting; heavier leaf-value shrinkage generalized much better.
3. **More trees, enabled by regularization.** `n_estimators=1000` (with early
   baselines of 300) (0.7241 -> 0.7247). Without the strong L2 this overfit.
4. **Diverse 8-model XGBoost ensemble.** Different `max_depth` (4-7), learning
   rate (0.015-0.04), subsample, colsample, and tree counts, averaged as
   probabilities (0.7179 single -> 0.7251). Diversity of configs helped; extra
   *seeds* of the same config did not.
5. **Origin/Dest x departure-hour congestion counts.** Log1p of train frequency
   of flights leaving each Origin (and arriving at each Dest) in each hour
   (0.7251 -> 0.7279). Captures airport busy-hour congestion, stable across years.

## What did NOT help

- **High-cardinality categoricals**: `route = Origin_Dest`, `origin_carrier`,
  `dest_carrier` as XGBoost categoricals dropped AUC to 0.70-0.71 (overfit under
  the year shift). Keeping Origin and Dest separate is strictly better.
- **Target encoding** of route/origin/dest/carrier (smoothed) fell to 0.7102:
  the boosters leaned on the encoded training means and failed out-of-year.
- **Generic frequency encodings, rare-level grouping, `log_distance`**: all
  neutral to slightly negative; removed for simplicity.
- **Over/under-regularization and other knobs**: `reg_lambda=50/alpha=10`,
  `gamma`, `max_bin=128`, `colsample_bynode`, `min_child_weight` x3, DART, and
  rank-averaging were all <= the kept config.

## What I would try with more budget

Go after *stable cross-year* structure rather than more model capacity: add
time x airport/carrier interactions (e.g. carrier banks at an airport-hour),
cyclical day-of-year/seasonal encodings, and weather/holiday proxies if any
external-but-legal signal existed. A proper nested CV for choosing
regularization and ensemble weights (rather than eval-based comparison) would
reduce the risk of overfitting the eval year. A stacked meta-learner over model
outputs and a small hyperparameter search over regularization strength
(jointly with tree count) are the most promising next steps; the plateau at
~0.725 without congestion features suggests feature engineering, not tuning,
is the remaining lever.
