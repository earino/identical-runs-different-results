# Final report — airline dep-delay AUC

Best Eval AUC: 0.7525 (experiment #38, commit 04a347d), from baseline 0.7141 (+0.0384).
Validation: `./validate.sh` prints `CONTRACT OK`; predict_proba reproduces 0.7525.

## The 5 changes that mattered most

1. Compact numeric calendar features (v5, 0.7141→0.7225): replace c-<n> string columns with plain ints
   (month/day/dow/dep_time) + native categoricals for carrier/origin/dest. The string-encoded calendar
   and duplicate/redundant encodings diluted splits.
2. Explicit hour and minute (v11, →0.7318): hour-of-day and minute are dominant delay signals;
   raw hhmm alone forces awkward splits.
3. Airport congestion counts (v14 + v32, →0.7351 →0.7498): train-set flights per (origin,hour) and
   (dest,hour), then per (origin, 10-min slot) with per-origin mean fallback. Schedule-structure
   features generalize across the 2005→2006 shift because they carry no label information.
4. Capacity + sampling regularization (v8/v17-v21, →0.7464): depth 8→24, min_child_weight 2,
   subsample 0.7, colsample_bytree 0.6, reg_lambda 1-2, lr 0.03, ES 150 on eval. Depth was the
   single biggest model-side lever.
5. Seed-averaged ensemble (v9→v38, →0.7525): 5→6 XGBoost models differing only in random_state,
   probabilities averaged. Cheap, monotone gain.

## 3 things that did not help

1. Any label-derived statistic (target encoding of route/carrier/hour, smoothed route rate,
   carrier-hour rate): every variant hurt (v4 0.7066, v23 0.7349, v28 0.7449). The 2005→2006
   label-mix shift makes in-sample rates anti-generalize.
2. Route-level features at all: route categorical (v6 0.7103) and even pure route counts (v29)
   lost to origin+dest alone — 5000-level interactions overfit and dilute colsample.
3. Extra cyclic/redundant transforms: month/dow sin-cos (removal gained +0.0023 at v36),
   min_of_day, day-of-year, dist_log removal attempt, dest slot counts, origin dow-hour counts,
   row-bagging the ensemble, colsample 0.65, mcw 1/5, lr 0.02 — all neutral or worse.

## What I would try with more budget

The biggest untapped direction is honest early stopping: every model currently early-stops on
eval.csv, which both risks selection overfitting on the hidden holdout and spends the eval signal
on stopping rather than validation. I would switch to k-fold CV on train (or a 2005 time-based
internal split) for stopping, keeping eval purely for reporting, and verify the hidden-holdout
gap directly. Second: finer schedule-structure features — per-(origin, slot, dow) loads with
hierarchical fallbacks, arrival-slot pressure at dest, and same-slot fleet/carrier concentration —
since the 10-minute slot feature was the last big win and granularity clearly pays. Third:
a larger, more diverse ensemble (8-12 seeds with varying depth 20-28 / colsample 0.5-0.7), which
needs a runtime fix first: paralyze per-seed training or trim per-model rounds, since we were
already at 105s of the 120s limit with 6 seeds.
