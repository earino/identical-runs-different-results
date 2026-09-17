# FINAL — airline delay (XGBoost, hidden-holdout AUC task)

**Best Eval AUC: 0.7432** (baseline 0.7141 → +0.029). HEAD = `exp41`, validated: `CONTRACT OK`
(`validate.log`, AUC 0.7432 reproduced through `predict_proba` with the target column dropped).

## Changes that mattered most

1. **Big-capacity single model with heavy regularization** (exp7–exp9, kept in exp20/34/35): depth 12→13,
   lr 0.02, min_child_weight 25, subsample 0.85, colsample_bytree 0.35, lambda 8, gamma 2, ~2600 trees with
   early stopping (patience 80) on a random 15% of train. This was the single biggest jump (0.7167 → 0.7321);
   the deep-but-strongly-shrunk regime fits the delay interactions without overfitting the 100k rows.
2. **Time-of-day features** (exp2): DepTime decoded from raw hhmm (with the ≥2400 wrap) into minutes-since-midnight,
   cyclical sin/cos, and hour as a categorical. +0.0026 over the baseline that fed raw hhmm to the trees.
3. **Carrier × time interaction as a native categorical** (exp28, then exp38): carrier joined with hour, later
   refined to **half-hour buckets** — 0.7349 → 0.7425, the largest single-experiment gain of the run. Delay is
   strongly carrier-time-of-day dependent and the model uses the interaction directly.
4. **Airport × half-hour interactions** (exp30, kept at threshold 150+ train rows): origin/dest joined with
   half-hour buckets, frequent combos only → 0.7363.
5. **Final regularization nudge** (exp41): subsample 0.85 → 0.9 → 0.7432.

## What did not help

- **Out-of-fold smoothed target encoding** of carrier/origin/dest/route/airport-hour keys (exp5, exp6): 0.7238
  vs 0.7249 native — no gain, more machinery; removed.
- **High-cardinality native categoricals** (route, origin×hour at full cardinality, hour×dow, carrier×hour×dow):
  all at or below the best without them (0.7170, 0.7299, 0.7276...). Only *frequency-filtered* interaction keys
  (≥30/≥150 rows, or low-cardinality carrier/hour) helped.
- **Seed bagging** of two identical models (exp17/18): 0.7324 vs 0.7327 single — no gain for 2× cost.
- **Time-based early-stopping split** (exp12): 0.7310 vs 0.7321 random split (eval is 2006, but the 100k train
  slice appears not to be in strict time order, so the "time-based" split was just a worse random one).
- **Cyclical month/dow/dom sin-cos** (exp24): 0.7302 vs 0.7331 — month is better as a plain ordinal for splits.

## Theory of the data

Delay risk is dominated by scheduled departure time interacting with carrier and airport (banking schedules,
hub congestion waves) — hence interaction categoricals at fine time granularity dominate. Everything static
(calendar ordinal day/month, distance) is secondary. The eval slice is a different year, so robustness came
from frequency-filtered keys (unseen combos → NaN, handled natively) rather than leakage-prone target stats.

## With more budget

- Frequency thresholds as proper hyperparameters (sweep 30/60/100/150/300 per key, including a route key).
- Quantile-bucketed time (learned breakpoints) instead of fixed half-hours; 15-minute buckets for the carrier key.
- Small 3–5-model ensemble varying seed + colsample (bagging gave nothing with 2, but diversity via feature
  subsets might); stacking the exp38-style model with a shallower model on TE features.
- Full CV-fold early stopping (fit per-fold models and average) instead of a single 85/15 split.
