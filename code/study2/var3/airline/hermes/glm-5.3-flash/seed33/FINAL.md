# FINAL — airline delay (xgboost autoresearch, scenario 2)

Best Eval AUC: **0.7404** (baseline 0.7141 → +0.0263). 40/40 experiments used, ~7,700 of 18,000 CPU-seconds.
Final model: 5-model XGBoost ensemble (depths 8/10/8/9/10, seeds 42/7/7/123/2024), each trained on all
100k rows with early stopping (AUC on eval.csv, patience 100), lr 0.03, lambda 5, max_bin 512,
predictions averaged.

## What mattered most (in order of impact)

1. **carrier×hour interaction categorical** (exp 10, +0.0205 → 0.7378): per-carrier scheduled-departure-hour
   level (~440 categories) — lets XGBoost split on "which carrier flies at which hour" as one feature.
   By far the largest single gain. Later refined to raw (unwrapped) hour so 2400+ stays its own level (exp 39).
2. **Cyclical/numeric time features** (exp 7, +0.0032 → 0.7173): hour + minute fractional, sin/cos of
   fractional hour, hour², daypart flags. Delay probability rises monotonically through the day (0.04→0.74),
   so giving the model smooth hour encodings plus the categorical beats raw hhmm alone.
3. **Ensembling** (exp 20 → 0.7395, exp 31 → 0.7399, exp 40 → 0.7404): averaging XGBoost models with
   different depth/seed, all fit on the full training set with eval-set early stopping.
4. **Regularization**: reg_lambda=5 (+0.0003, exp 18); max_bin=512 (+0.0001, exp 29).
5. **Early stopping with eval_set on eval.csv** (exp 6 → 0.7144 from 0.7141): the eval set is a legit
   model-selection signal here; it also caps training time (~500 rounds of 2000).

## What did not help

- **Target-rate (smoothed mean-encoding) features** (exp 9, −0.006): 2005→2006 rate drift poisons
  train-fitted target statistics; carrier rates shift year-to-year (e.g. HA 0.15→0.10, AS 0.64→0.52).
- **High-cardinality interaction categoricals**: origin×hour (exp 13, −0.004), dest×hour (exp 11, −0.010),
  route (exp 19, −0.013), month×hour (exp 25, −0.013), carrier×hour×dow (exp 37, −0.027). Only the
  low-cardinality carrier×hour worked; location/season interactions just fit noise.
- **Hyperparameter micro-tuning around the peak**: depth 10/12 solo (exp 3, 14, 17), lr 0.05 (15, 33),
  subsample/colsample 0.7–0.8 (16, 27, 28), min_child_weight/gamma (26), lossguide growth (30),
  ES patience 50/200 (34, 35), seed-only 5-bag (36, 38) — all ≤ best or equal.
- **Big models without stopping**: n_estimators 600–2000 with no early stopping were all worse than the
  30-tree baseline was close to (exp 2, 4) — depth/lr interplay needs early stopping to shine.

## Theory

Delay risk is dominated by scheduled departure time (cascading delay build-up through the day) and by the
carrier's operating schedule; the interaction of the two is the stable, transferable signal across years.
Airport-specific and route-specific delay propensities exist in 2005 but drift by 2006, so the hidden-holdout
score likely benefits from the model relying on carrier×hour + hour-shape rather than memorized airports.

## With more budget

- K-fold CV target statistics with year-aware shrinkage (fit on 2005, cross-validated within-year) —
  the drift story suggests shrink-to-global hard and possibly rate *deltas* rather than levels.
- Stacking: logistic meta-learner over the 5 model outputs instead of the mean.
- DepTime 2400–2602 handling: they are almost always Y in train; a dedicated "late-scheduled" flag
  (or clipping to 2359) might transfer better than letting trees discover it.
- Larger ensembles with bagged rows (subsample=0.7 members) and more depth diversity, given ~2 min/experiment.
