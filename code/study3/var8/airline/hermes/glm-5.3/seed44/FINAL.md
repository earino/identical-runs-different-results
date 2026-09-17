# FINAL — airline dep_delayed_15min (XGBoost)

Best Eval AUC: **0.7211** (baseline: 0.7141, +0.0070). Final model: commit `1b63d8a` ("exp39"),
validated with `./validate.sh` → `CONTRACT OK`, predict_proba AUC 0.7211.

## What mattered most (kept)

1. **Regularization package at modest capacity** (exp6): subsample 0.7, colsample_bytree 0.7,
   min_child_weight 30, reg_lambda 2, depth 5, 150 trees @ lr 0.08 → 0.7153 from 0.7141 baseline.
   The 2005→2006 time shift punishes variance hard; capacity increases always hurt.
2. **Congestion-volume features** (exp19/21): train-fitted flight counts per (Origin, hour),
   (Dest, hour), (Carrier, hour) plus marginal origin/dest volumes → 0.7188. Physical and stable
   across years, unlike any target-rate feature.
3. **Seed ensembling + rank averaging** (exp10→26): 12→24 models, probability mean replaced by
   mean of rank-normalized scores → 0.7190.
4. **Grow-policy family diversity** (exp29): blending depthwise-5 with lossguide (max_leaves 24)
   ensembles beat pure seed scaling (0.7191 vs 0.7189 at 48 pure seeds).
5. **Aggressive colsample walk + relaxed leaf constraints** (exp32→36, 39): colsample 0.25/0.4/0.2,
   min_child_weight 5, reg_lambda 0.2, third family (depth 6 @ colsample 0.2) → 0.7205 → 0.7211.

## What did not help (discarded)

- **Capacity / same-year early stopping** (exp2, 4, 7): depth 6-8 or 200-300 trees at depth 6
  always dropped AUC; early stopping on a 2005 split locks in 2005 noise.
- **Rich feature engineering** (exp3, 5, 8): route strings, month/day numeric twins, cyclic
  sin/cos of time, log distance — all neutral-to-harmful; the tree already reads raw DepTime.
- **Target encoding of any kind** (exp16, 20): smoothed TE of carrier/origin/dest/route
  (0.7082) and smoothed delay-rates per (origin, hour) (0.7128) — 2005 delay rates drift badly
  to 2006; volume counts are stable, delay rates are not.
- Also no-ops/negative: monotone constraint on hour (exp27, red-eye hours are non-monotone),
  recency sample weights (exp31, exact tie), route-hour counts (exp23, too sparse),
  48 seeds (exp28, saturated), d3 third family (exp30).

## With more budget

The two biggest untested levers: (1) systematic hyperparameter search around the exp39 point
(colsample 0.15-0.5, mcw 1-10, per-family tree counts, max_bin, gamma) — the colsample/mcw walk
was still monotonically improving when budget ran out; (2) richer *volume-only* lookups that
 stayed untested: origin×hour×dayofweek, seasonal month-volume, and number of departures in the
same hour bucket ±1 hour at the origin (rolling congestion window). I would also try blending
in a model trained on hour-of-day only (the most stable signal) as a feature-level prior, and
re-examine whether eval-robust gains (≥0.0005) from a 5-fold time-blocked CV would transfer
better than my single eval.csv decisions — with one eval slice, gains under ~0.0003 were kept
only when part of a consistent trend.
