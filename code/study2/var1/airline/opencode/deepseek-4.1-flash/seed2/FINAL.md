# Final report — airline departure-delay AUC

**Best Eval AUC: 0.7500** (commit `928924e`, experiment #24; validated: `CONTRACT OK`).

## Task
Binary classification on the airline dataset (predict `dep_delayed_15min`), trained on 2005 and
evaluated on 2006. Metric: ROC AUC. Final model is an XGBoost-only ensemble with all feature
engineering inside `prepare(df)` so it reproduces on the hidden holdout.

## Changes that mattered most
1. **Shallow + deep XGBoost ensemble.** Averaging two shallow trees (`max_depth` 2, 3) with two
   deep trees (`max_depth` 20, 32, weighted x2) beat either family alone by ~0.014. The shallow
   members capture stable global structure; the deep members capture interactions. This is the most
   robust change against the 2005→2006 distribution shift.
2. **Strong column subsampling on the deep members** (`colsample_bytree=0.6`). This regularizes the
   deep trees (a single deep model dropped from ~0.74 to ~0.73 without it) and is essential for them
   to generalize across years.
3. **15-minute time-of-day categorical** (`dep_tod // 15`, 96 levels). The single largest feature
   win: 0.7436 → 0.7500. A native categorical lets the tree split on exact scheduling slots instead
   of approximating them with axis-aligned cuts.
4. **Calendar features**: day-of-year, weekend flag, and cyclical day-of-week sin/cos.
5. **Weighted probability averaging** (deep members x2, shallow x1, logit-free mean).

## Things that did not help (all reverted)
- High-cardinality route/carrier-route categoricals and smoothed target / frequency encoding.
- Holiday flags, cyclical hour/month encodings, log-distance and time×day interaction features.
- Coarser (hour, half-hour) or finer (10-min, 5-min) time-of-day bins than 15 minutes.
- Early stopping on a train holdout, row subsampling (`subsample=0.9`), logit-space averaging.
- `dart`/`lossguide` boosters, `min_child_weight`/`reg_lambda`/`max_bin`/`max_cat_threshold` tuning.
- One-hot / medium-depth members, extra seed bagging, additional deep depths (d50/d64).

## What I would try with more budget
Out-of-fold **stacking** with a small XGBoost meta-learner (base-model OOF predictions per
time-of-day / carrier segment) could beat plain averaging, but the deep members are too slow to
generate enough folds within the 120 s experiment cap. **Adversarial validation** (train-vs-eval
classifier over features only) to drop or down-weight year-drifting features is another promising,
label-free way to improve hidden-holdout generalization. Finally, a week-of-year categorical and
per-carrier time-of-day categorical interactions are untested feature directions that follow
directly from the 15-minute-bin result.
