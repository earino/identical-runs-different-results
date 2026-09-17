# FINAL — airline delay XGBoost

**Best Eval AUC: 0.7451** (experiment 35, commit `f47ccac`, "categorical hour")

## Setup
XGBoost `XGBClassifier` (hist, 4 threads, `enable_categorical=True`), trained on
`data/train.csv` (2005), evaluated on `data/eval.csv` (2006). All feature engineering lives in
`prepare()` / `add_features()` and is fit on training data only, so `predict_proba(df)` reproduces
it on unseen rows.

## Changes that mattered most
1. **Parsed time features** (exact `hour`, `minute`, `dep_min` from the `hhmm` `DepTime`) plus cyclic
   `hour_sin/cos`, `dow_sin/cos`. Biggest feature-engineering win (0.714 → 0.720).
2. **Tree capacity** — `max_depth` from 6 up to 20 and `n_estimators` 30 → 1500 with `learning_rate`
   0.1 → 0.015. The baseline was heavily underfit (0.720 → 0.734).
3. **Feature subsampling `colsample_bytree=0.5`** — the single largest jump (0.734 → 0.741); strong
   per-tree feature dropout regularized the deep-tree model.
4. **`max_cat_threshold=256`** — lets categorical splits of Origin/Dest/UniqueCarrier consider more
   levels (0.7413 → 0.7417).
5. **Categorical `hour_cat`** (hour as a 24-level categorical alongside numeric hour) captures
   arbitrary/cyclic hour groups and was the final lift (0.7417 → 0.7451).

## What did NOT help
- **Out-of-fold target encoding** of Origin/Dest/UniqueCarrier/Route (0.7313) — redundant with
  XGBoost's native categorical handling and added noise.
- **Route as a high-cardinality categorical** (0.7174) or carrier×hour interaction — overfits and/or
  blows the wall-clock budget.
- **Strong regularization / extra features** — `min_child_weight=10` (0.7248), extra seasonal
  (`is_weekend`, `doy_sin/cos`, `log Distance`) (0.7344), categorical month/dow/minute, `max_bin=128`,
  `lossguide`, DART (timeout), and frequency encodings (neutral) all failed to beat the best.

## What I would try with more budget
Push the capacity/regularization frontier more systematically: use a proper internal validation split
(there is no room to use eval.csv for model selection without risking hidden-holdout overfitting) to
tune `n_estimators`/`eta`/`colsample_bytree` jointly, and build a seed/feature-subsample ensemble of
the best config — averaging several XGBoost models was neutral on eval but is the most promising
low-variance route to a robust hidden-holdout gain. I would also explore finer time encodings
(e.g. departure-time buckets interacted with Origin) with time-aware regularization, since the
hour-categorical result shows time-of-day grouping is the key remaining signal.
