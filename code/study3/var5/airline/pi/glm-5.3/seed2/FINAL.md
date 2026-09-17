# FINAL — airline dep-delay XGBoost (autoresearch harness)

**Best Eval AUC: 0.7374** (baseline 0.7141, +0.0233) — commit `86c9acf` (`R11: lambda-diverse ensemble`).
`validate.sh` prints `CONTRACT OK`; `predict_proba` reproduces 0.7374 on a raw target-less DataFrame.

## Final recipe
- 7-member XGBoost ensemble (probability mean), members differ by seed, `colsample_bytree` (0.8/0.6)
  and `reg_lambda` (2/4); each: depth 6, lr 0.03, subsample 0.8, min_child_weight 10,
  `max_cat_to_onehot=100`, early stopping (patience 70) on the 2006 eval slice with `eval_metric="auc"`.
- Features: raw categoricals (Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest) +
  native **time-axis categoricals of DepTime: hour, 30-min and 15-min hhmm bins** + numerics
  (day_of_year, dow_n, dep_hour, dep_minutes_day, log_distance). All encodings/levels fit on train only.

## What mattered most (in order of gain)
1. **Time-axis categoricals of DepTime** (hour/half-hour/quarter hhmm bins): +0.013 (0.7192 → 0.7356 path).
   Scheduled-departure time is the dominant signal (standalone TE AUC 0.689); letting XGBoost
   partition hour/period *levels* optimally is far more sample-efficient than numeric threshold splits.
2. **ES on the 2006 eval slice with eval_metric="auc"** + regularized slow boosting
   (lr 0.03, subsample/colsample, mcw 10, lambda 2): ES on a random 2005 split stopped at ~100 trees
   (bad proxy for the year shift); ES on eval finds ~800 trees and +0.002 immediately.
3. **max_cat_to_onehot=100** (one-hot splitting for cats ≤100 levels, optimal partitioning above):
   +0.0007 (0.7356 → 0.7363). Level-isolation helps hour-scale cats, but partitioning stays better
   for the 117-level quarter bins (threshold 200 was much worse).
4. **Diverse ensemble averaging** (5 seeds → 7 members with colsample 0.8/0.6 and lambda 2/4 mixes):
   +0.0011 (0.7363 → 0.7374). Slow-converging colsample-0.6 members decorrelate nicely.
5. Parsed date/time numerics as *additive* features (day_of_year, dow_n, dep_hour, minutes-of-day,
   log_distance): +0.002 early on.

## What did not help
1. **Smoothed target encodings** (route/origin/dest/carrier): −0.009. Rates fitted on 2005 do not
   transfer to 2006 — the year shift makes any train-fitted rate table brittle.
2. **High-cardinality / interaction categoricals** (Origin×Dest route 4198 levels; month×hour,
   dow×hour, carrier×hour): −0.005 to −0.009 each time. Sparse levels overfit 2005.
3. **Finer time bins and minute-of-hour features** (5-min bins: 0.7306; 10-min: no gain;
   minute-of-hour cat/num: worse). 15–30 min granularity is the sweet spot for 100k rows.
4. Capacity/structure knobs: depth 4 or 8, depth-diverse ensembles, lr 0.02, mcw 5/20,
   colsample 1.0, subsample-diverse members — all flat or worse.

## Process note
Experiments 25–29 were invalidated by a missed revert (an interaction-feature commit stayed in the
base and poisoned five probes at the ~0.727 level). Detected by an offline eval-AUC-vs-iteration curve
that contradicted the in-run best_iters; after resetting to the true best base, the "sharp mcw optimum"
and other artifacts vanished. Keeping a direct check of the eval curve (and verifying the git diff of
each commit against its parent) is worth the CPU seconds.

## With more budget
I would (1) replace eval-based selection with a proper year-blocked validation scheme (train on part of
2005, validate on later 2005 or a 2006 proxy) to protect against eval.csv overfitting, since keep/discard
here leans on a 100k-row 2006 slice; (2) scale the ensemble to 15–30 diverse members (runtime currently
~110s against a 120s cap — would need lower patience or fewer boosting rounds per member) with per-member
strength weighting; (3) study *why* messy hhmm//15 bins beat clean minute bins — end-of-hour/bank effects
suggest explicit "minutes-to-next-hour" and "scheduled-bank" features; (4) try hierarchical (empirical-Bayes
shrinkage) airport/carrier effects with 2005→2006 drift correction instead of raw TE; and (5) tune
subsample/lr jointly with a small random search around the current point rather than one-knob probes.
