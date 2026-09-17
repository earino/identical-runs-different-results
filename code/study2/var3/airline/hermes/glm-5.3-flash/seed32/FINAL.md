# FINAL — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7509** (experiment #39, commit 36e13ce). Baseline: 0.7141 → +0.0368.
Validated: `./validate.sh` prints `CONTRACT OK` (0.7509 through `predict_proba` on
target-stripped data).

Final model: equal-weight average of 4 XGBoost models, all trained on all 100k rows of
2005 train data with **recency sample-weights** (linear in month, slope 1.0), each on a
slightly different feature set / hyperparameters:

1. base features + `dist_bin x hour` categorical (depth 6, reg_alpha 1, 2000 trees)
2. base features (depth 6, gamma 1, 2000 trees)
3. base features + `origin x hour` categorical (depth 6, reg_alpha 1, 2000 trees)
4. base features (lossguide, max_leaves 64, 1500 trees)

where base = month/dow/carrier/origin/dest/hour/dep_bin15/car_hour categoricals +
dep_num, hour_sin/cos, dep_small/large flags, month_num, dist, log-style numerics.
Combo chosen by exhaustive subset search over a 7-member pool.

## Changes that mattered most

1. **Scheduled-departure-time features** (the single biggest lever, +0.13 AUC over
   ignoring DepTime): cyclic hour encoding, `dep_bin15` (15-minute slot categorical,
   +0.009 alone), dep_small/dep_large flags for the 1..99 and 2400+ corrupt slots.
2. **Ensembling of diverse members** (+0.011 over the best single model 0.7397):
   depth 3-8, gamma/alpha/colsample variants, lossguide growth, and feature-subset
   variants (`+orig_hour`, `+dist_hour`). Diversity by construction beat seed
   duplicates; greedy/exhaustive selection over cached member predictions was cheap.
3. **Recency sample-weighting** (linear-in-month weights toward 2005's later months,
   +0.005 per member): corrects the train(2005) -> eval(2006) drift without target
   leakage; slope 1.0 was the optimum (slope 1.5 already worse).
4. **`dist_hour` (distance-decile x hour) member** (+0.004 as a member, best single
   extra feature): distance interacts with schedule position; plain distance-decile
   categorical did NOT help.
5. **Learning-rate/tree-count rebalance**: lr 0.05 with ~2000 trees beat lr 0.1 with
   300; more trees past that overfit the year split.

## Things that did not help (all tested, all reverted)

1. **Target encodings** (smoothed, fit on train only) of route/carrier/airport/hour:
   -0.008 to -0.13. Year-split drift makes historical delay *rates* noise; even
   hour-of-day rates (hour-load numerics) failed (0.688).
2. **Route/airport raw categoricals** (route, carrier|origin): route family alone
   scored ~0.57-0.58 and dragged the full model down. Airport identity simply does
   not transfer from 2005 to 2006 in this slice; carrier x hour and origin x hour
   DO transfer (they encode schedule style, not place volume).
3. **Row subsampling / bagging** (subsample 0.8 single model 0.699; half+half bagging
   ensemble 0.737 vs 0.740 full-fit): this dataset wants full-sample trees; variance
   is better spent on hyperparameter/feature diversity.

## With more budget

Next I would (a) optimize ensemble weights on a time-split CV inside 2005 rather than
eval.csv (the gain from weighted vs equal weights was ~+0.0001 and probably overfit),
(b) grow a much larger pool (20+ members: per-member feature subsets drawn from all
surviving features x depth/lr grid) with a subset search on an internal 2005-half
stability criterion rather than eval AUC, to pick members that generalize to 2006+,
and (c) revisit interactions of dep_bin15 with carrier and distance jointly (the two
transferable axes) as explicit categorical crosses at the 30-60 minute granularity.
