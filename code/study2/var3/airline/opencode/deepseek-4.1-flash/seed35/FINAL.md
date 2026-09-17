# Final Report — airline XGBoost AUC

**Best Eval AUC: 0.7209** (experiment #40, commit `e2215f5`), vs baseline 0.7141.
Contract validated: `./validate.sh` prints `CONTRACT OK` (eval AUC via `predict_proba` = 0.7209).

## What the final model does

`prepare(df)` is the single feature path used for both training and `predict_proba`:

- Raw categorical columns (`Month`, `DayofMonth`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest`) as
  XGBoost native categoricals with levels learned from `data/train.csv` only.
- Out-of-fold smoothed target encoding (`sklearn TargetEncoder`, `smooth=50`, `cv=10`) for
  `UniqueCarrier`, `Origin`, `Dest`.
- Departure-time features from `DepTime`: `dep_min` (minutes since midnight), `dep_hour_sin/cos`,
  `is_weekend`.
- `Distance` numeric.
- An ensemble of 8 `XGBClassifier` models (depth 3–7, several seeds) with strongly regularized
  lossguide trees (`max_leaves=48`, `max_bin=128`, `subsample=0.6`, `colsample_bytree=0.5`,
  `min_child_weight=10`, `gamma=3`, `reg_alpha=3`, `reg_lambda=20`, `lr=0.05`, 300 trees).
  Probabilities are averaged.

## Changes that mattered most

1. **Strong regularization** — the biggest lever. Adding `gamma/reg_alpha` on top of a regularized
   base moved eval from ~0.718 to 0.7187→0.7202. The 2005→2006 time shift makes the model overfit
   train quickly; heavier `gamma`/`reg_alpha`/`reg_lambda` was worth far more than capacity.
2. **Out-of-fold target encoding** of carrier/origin/dest (`+0.0021` over baseline at the time).
3. **Seed/depth ensemble averaging** (`+~0.001`): variance reduction that transfers to unseen data.
4. **Departure-time features** — neutral early, but with strong regularization they finally helped
   (`+0.0006`): `dep_min`, hour sin/cos, `is_weekend`.
5. **lossguide + `max_bin=128`** and structural params (`max_leaves=48`) gave a small consistent bump.

## Things that did NOT help (reverted)

- **Route target encoding** (`Origin_Dest`) — consistently −0.001 to −0.002; too high-cardinality.
- **Target encoding of Month/DayOfWeek or dep_hour**, and **frequency encoding** — neutral-to-worse.
- **DART booster** — timed out (>120 s), discarded.
- **Bigger/more diverse ensembles or more trees** (500–600, lr 0.03) — no gain once regularized.
- Raw feature removal / cycling transforms at baseline (pre-regularization) hurt.

## With more budget

The eval differences here are only a few 1e-4, so most micro-gains are within noise; the robust
levers were regularization, target encoding and ensembling. More budget would go to (a) a proper
time-aware validation split (train on early 2005, validate on late 2005) to choose `gamma` and
ensemble size against a less noisy proxy, (b) bagged target encodings across multiple random folds
and smoothing levels, and (c) a careful search over `max_leaves`/`colsample` interactions. Weather
and aircraft-tail data are not in the schema, which caps the achievable AUC on these eight columns.
