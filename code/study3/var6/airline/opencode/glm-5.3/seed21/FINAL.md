# Final Report

**Best Eval AUC: 0.7486** (experiment #27, commit `0cad26a`), up from the 0.7141 baseline.

## The 5 changes that mattered most

1. **Out-of-fold target encoding with fine time granularity** (the core signal): smoothed delay-rate
   TEs for carrier/origin/dest/route crossed with departure time at progressively finer bins —
   hour → 30-min → 15-min → 10-min slots (`te_r_15` with m=300 and `te_o_10`/`te_d_10`/`te_c_10`
   with m=50 became the strongest features). 10-fold OOF on the train side, full-train maps for
   serving, per-key smoothing constants. This ladder carried AUC from ~0.719 to ~0.735.
2. **Extreme feature subsampling (`colsample_bytree` 0.5 → 0.08) with low `min_child_weight` (20)**:
   switching to many decorrelated weak trees (d8, lr 0.02, ~230 boosting rounds) was the single
   biggest jump — 0.7362 → 0.7486 in two experiments.
3. **Slow, heavily-regularized boosting** (lr 0.02 + early stopping on eval) instead of the default
   fast profile: +0.001-0.002 early on and it stayed optimal all the way through.
4. **Numeric time/calendar features** added alongside the TEs (`dep_minutes`, `dep_hour`, sin/cos of
   time-of-day and month, day-of-year, num dom/dow/month): +0.0015, and they let trees model the
   global delay ramp the entity TEs can't.
5. **Seed-bagged 5-model ensemble with per-model TE fold seeds**: each member sees a different OOF
   fold-noise realization of the TEs, decorrelating both tree randomness and feature noise
   (+0.0007-0.001, and robustness for the 1M-row holdout).

## 3 things that did not help

1. **More capacity / alternative tree shapes** once features were strong: d10/d12, mcw 300-400,
   gamma, lossguide, lr 0.01/0.03, subsample 0.5/0.6/0.85 — all flat or worse; capacity peaked at
   d8 and the regularization role moved entirely into colsample.
2. **Pruning weak features**: dropping the 6 lowest-importance features *hurt* (0.7364 → 0.7357);
   even weak columns carry split-support value in the low-colsample regime.
3. **Everything beyond the core TE family**: route×10-min TEs (too sparse), carrier×airport pair
   TEs, global month×slot / dow×slot delay rates, flight-volume counts, time-decay sample
   weights, TE contrasts/hierarchical smoothing — all flat or worse. The TE design was saturated;
   the remaining headroom was in the model profile, not more features.

## What I would try with more budget

Coordinate-descent the whole weak-learner profile jointly (colsample/subsample/lr/depth/mcw/rounds
interact, and 0.08/20 was found in the last 20 minutes of budget), including even lower
colsample with more boosting rounds. Then diversify the ensemble at the *model* level: members
with different colsample/depth/feature subsets (feature-bagged ensemble), and 8-10 members —
in the extreme-decorrelation regime each member is weak, so averaging should keep paying.
Finally, OOF stacking of that diverse base with a logistic/XGB meta-learner for the last
+0.001-0.002. On the feature side I would try "airport daily-profile" TEs (origin's delay rate
early vs late in the same day, learned from 2005 schedules) which is the one interaction the
current TE family does not encode.
