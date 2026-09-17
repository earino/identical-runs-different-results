# FINAL — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7512** (baseline 0.7141, +0.037). Final commit: `bbd5957` (reg_alpha 2.0), validated `CONTRACT OK`.

## Final model

Ensemble of 6 XGBoost classifiers (mean of predicted probabilities), all trained on the full
`data/train.csv` with `hist` tree method and native categorical split support:

- members: `(max_depth, colsample, subsample, rounds)` = (16, .6, .7), (18, .7, .75), (20, .8, .8),
  (20, .9, .85), (22, 1.0, .9), (24, .85, .8), all 300 rounds, lr 0.05, min_child_weight 5,
  reg_lambda 2, reg_alpha 2, seeds 1/7/42/123/2024/555.
- features: Month/DayofMonth/DayOfWeek as ints, hour-of-day + 24-level hour_cat, tod sin/cos,
  Distance + log1p, carriers/Origin/Dest as categoricals with train-fitted levels. All engineering
  lives in `prepare(df)`; encoders fit on train only.

## Changes that mattered most

1. **Feature engineering v1** (0.7141 → 0.7232): cyclic time-of-day encoding, hour categorical,
   log-distance; early stopping on an internal split + refit on full train.
2. **Regularized deep single model** (→ 0.7330): min_child_weight 5, subsample/colsample 0.8,
   reg_lambda 2, and increasing max_depth — depth 20 was the sweet spot (24 overfit).
3. **Pruning weak cyclic features** (0.7375 → 0.7451): removing month/day-of-month/day-of-week
   sin/cos raised AUC by +0.0076 — they diluted the colsample pool and invited noise splits.
   Time-of-day sin/cos ablation confirmed those ARE useful (removing them cost −0.006).
4. **Ensembling** (→ 0.7375 before pruning): 6 members beat 5 and 4; per-member colsample and
   subsample diversity helped. 4 members × 450 rounds lost to 6 × 300 — member count > per-member fit.
5. **L1 regularization sweep** (0.7452 → 0.7512): reg_alpha 0.5 → 1.0 → 2.0 gave monotone gains,
   the largest single-param win of the run (+0.006 total).

## Things that did not help

- Target encoding in all variants (plain smoothed, route-only, out-of-fold): always below the
  no-TE baseline; native categorical splits already capture level effects.
- Interaction categoricals (carrier×hour, origin×hour, month×dow, 15-min tod bins): diluted the
  sampled feature pool and hurt, despite being the "classic" airline-delay trick.
- lossguide growth (max_leaves 512), max_bin 512 (timeout), row bagging (80% per member),
  logit-space averaging, rank averaging, lr diversity, asymmetric round allocation.

## With more budget

I would (a) run per-member early stopping to calibrate rounds (~690 was ES-optimal at d16/d20 but
members get 300 — a faster fit path, e.g. `xgb.train` on shared DMatrix, would free the budget),
(b) grow the ensemble to 8–10 members with the saved time, and (c) explore DepTime×carrier and
route-level features with careful OOF encoding now that feature pruning showed the pool matters.
