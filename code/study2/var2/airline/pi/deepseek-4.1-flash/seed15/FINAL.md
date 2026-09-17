# FINAL — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7366** (experiment #39, commit `68ee401`), up from the 0.7141 baseline.
HEAD is the best `train.py` and `./validate.sh` reports `CONTRACT OK`.

## What the final model is

A seed-bagged ensemble of 14 XGBoost models (7 diverse configurations × 2 seeds) whose
predicted probabilities are averaged. All feature engineering lives inside `prepare(df)`, so
`predict_proba` reproduces it on unseen rows. Categorical levels are fitted on `data/train.csv`
only; unseen levels become NaN.

## Changes that mattered most

1. **`carrier_hour` interaction categorical** (`UniqueCarrier` × scheduled departure hour) —
   by far the biggest single win: 0.7218 → 0.7305. Carriers clearly operate different hub banks
   at different times of day, and an explicit interaction lets shallow trees use it.
2. **Model averaging (ensemble)** — 5 diverse configs (depths 3–6, learning rates 0.03–0.05,
   differing `min_child_weight`/`colsample`/categorical settings), 2 seeds each, averaged:
   0.7200 → 0.7215+, and it made aggressive single-model choices much safer.
3. **Many more trees per member** once `carrier_hour` was present — monotone gains from 300 up to
   ~1200–1800 trees (0.7305 → 0.7366), thanks to the extra signal and the averaging that
   regularizes it.
4. **Time features**: `dep_hour`, `dep_min`, `is_weekend`, day-of-year approx (`doy`), and
   `hour_sin`/`hour_cos` — a small but consistent gain (0.7174 → 0.7199).
5. **Shallower, mildly regularized base learners** (depth 4 rather than the baseline's 6 with few
   trees) — the first real improvement (0.7141 → 0.7174) on a dataset with limited generalizable
   signal and a year-to-year shift (2005 → 2006).

## What did NOT help

1. **High-cardinality interactions** — `route` (4198 levels, −0.016), `origin_hour`/`dest_hour`
   (−0.011), and coarse `origin_tod`/`dest_tod` (−0.003). They overfit the 2005→2006 shift.
2. **Target encoding** of Origin/Dest/Carrier/route (smoothed, out-of-fold) — neutral/slightly
   worse (0.7174 vs 0.7181).
3. **Sampling/regularization knobs** — `subsample`/`colsample` 0.8–0.9, `reg_lambda` 3–5,
   `min_child_weight` 20, and `grow_policy="lossguide"` all hurt.

## With more budget

The ceiling is set by the features: schedule-level information (carrier, airports, time, distance)
cannot capture weather or upstream aircraft rotations that actually cause delays, and the split is
a full year apart. I would (a) build **out-of-fold stacking** — feed member predictions plus a few
raw features into a level-2 XGBoost — instead of a plain average; (b) search finer carrier/time
interactions (e.g. carrier × 2-hour bins, carrier × day-type) with out-of-fold target encoding to
control cardinality; and (c) tune per-member tree counts and averaging weights via a train-internal
time-respecting validation split rather than picking them off `eval.csv`, which is the main
overfitting risk in the current keep/discard loop.
