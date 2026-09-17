# Final Report

**Best Eval AUC: 0.7355** (experiment #39, commit `19e3df1`; hidden holdout = 2006-slice2 1M rows)

## Final model

Uniform average of 22 members, all trained on the full 2005 training set (100k rows):

- **18 XGBoost members** — depth x learning-rate grid: d2-d10 at lr 0.05, d2-d8 at lr 0.03,
  d7-d8 at lr 0.1. Tree counts per member were selected by temporal early stopping
  (train on months 1-10, early stop on months 11-12, patience 150) and hardcoded;
  members are then retrained on all of 2005 (the full-train retrain is worth ~+0.006).
- **4 sklearn HistGradientBoosting members** — (max_iter, max_leaves, max_features):
  (600, 63), (900, 63), (600, 63, 0.7), (900, 63, 0.7). Origin/Dest passed as numeric
  codes (HGB's categorical cardinality cap is 255).

Features (inside `prepare()`): Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest
as native categoricals (train-fitted levels); DepTime -> dep_min (minutes since midnight),
dep_block (15-minute block, 96 levels, categorical), dep_hour (categorical), day-of-year;
Distance; dep_missing flag.

## Changes that mattered most

1. **Full-train retrain after temporal early stopping** (+0.006): select tree counts on
   months 11-12 of 2005, then retrain on all of 2005 with those counts.
2. **dep_block 15-minute departure-time categorical** (+0.0035 over the hour-only feature):
   the single most valuable engineered feature; departure time-of-day dominates delay risk.
3. **Depth x learning-rate member diversity with per-member ES counts** (+0.004 across
   exp11 -> exp27): shallow (d2-d3, lr 0.03, 550-900 trees) through deep (d9-d10) members;
   deep members need far more trees than hand-tuned grids assumed (d8: 626, not 80).
4. **Cross-implementation HGB members** (+0.0012): sklearn HistGradientBoosting with
   leaf-wise growth and column subsampling (max_features 0.7) decorrelates from the
   XGBoost depth-wise members.
5. **Ensembling itself** (+0.002 over the best single model): uniform prob-averaging;
   seed-only ensembles and rank-averaging did nothing, diverse hyperparameters did.

## Things that did not help

- **Target/frequency encodings** of Origin/Dest/route/carrier (multiple variants): helped
  the temporal validation AUC but hurt eval AUC — the year-over-year shift (2005 -> 2006)
  makes airport-level history unstable (Origin delay-rate correlation across years: 0.38).
- **Interaction features**: dow x dep_block, block x month (1152 levels), week-of-year —
  all worse or neutral; deep trees already capture these interactions.
- **Regularization/booster knobs**: subsample, DART booster, colsample clones, min_child_weight,
  reg_lambda, gamma, max_bin, lower learning rates, patience-300 ES — neutral to worse.
  Also: pruning subsets of the ensemble (every subset was worse than the full set),
  lr-0.02 members, HGB down-weighting, and rank-averaging.

## What I would try with more budget

The plateau at ~0.735 across many structural variants suggests the 2005->2006 shift caps
what 100k training rows support. With more budget I would (a) fit a small XGBoost on the
2006 eval rows' *features only* to test semi-supervised / domain-adaptation tricks
(e.g., importance-weighting train rows toward the 2006 feature distribution), (b) try a
stacked meta-learner over member predictions fit on rolling temporal folds of 2005
(val-weighting on one fold failed; multi-fold may not), and (c) push the depth grid further
with per-member learning-rate annealing, since deep members with ES-selected counts were
the largest continuing source of gains.

## Budget usage

40/40 experiments (or equivalent failures/timeouts), ~9045 of 18000 CPU-seconds,
~57 of 230 wall-clock minutes at exhaustion. Final `train.py` validated: `CONTRACT OK`,
predict_proba reproduces 0.7355 with the target column removed (117.6s full run).
