# Final report — airline dep_delayed_15min (XGBoost)

**Best Eval AUC: 0.7253** (baseline: 0.7141, +0.0112). Contract validated (`CONTRACT OK`).

## Changes that mattered most
1. **Target encoding** (exp7/36): smoothed, 5-fold out-of-fold TE of UniqueCarrier, Origin, Dest, and the
   Origin_Dest route, computed on train only (m = 25/50/50/50). Injects label statistics trees can't
   learn efficiently from 100k rows: +0.0026 alone, foundation for everything later.
2. **Day-of-year proxy** `30*(month-1)+day_of_month` (exp7; ablated in exp19, -0.002): captures
   seasonality (winter/holiday delays) that raw month/day splits reach only with extra depth.
3. **Leaf-wise trees** `grow_policy="lossguide", max_leaves=256` (exp25-28): replaced depth-5 symmetric
   trees; asymmetric trees fit the dominant time-of-day/seasonal patterns much better: +0.0010 to +0.004
   cumulatively, the single biggest structural win.
4. **Bagged ensemble of 10 XGBoost models** (exp13-14): seeds + subsample 0.8 force diversity,
   probabilities averaged: +0.002. Robust for the hidden holdout.
5. **Low learning rate 0.03 with early stopping** calibrated on eval.csv (exp5/15) + colsample 0.4
   (exp20-23): decorrelated members and precise round selection, each worth a few ten-thousandths.

## What did not help
- Tree-redundant features: dep_min/dep_hour (monotone in raw DepTime), log(Distance), numeric
  month/dow — pure dilution (exp4, exp6). Raw route categorical also hurt (exp6).
- Depth > 5 (exp2, exp10: 0.713) and interaction TEs (origin@hour, carrier@hour, origin#dow,
  carrier#month — exp11, exp33, exp40): too sparse under the 2005→2006 shift.
- Dropping raw high-cardinality categoricals in favor of TEs alone (exp35: -0.009) — TEs complement,
  not replace, native categorical splits.
- max_bin=64 (exp29: timeout; exp30: 0.7216) — fine thresholds matter for DepTime/doy/TE.
- min_child_weight, reg_lambda, 10-fold OOF, architecture blending (exp16/32/34/37/39): all neutral
  or worse; simplicity kept.

## With more budget
The leaf-count curve (32→64→128→256 leaves) was still rising when the 120s/run limit stopped it.
I would (a) raise max_leaves to 512-1024 while cutting rounds via warm-started lr schedules, exploiting
the CPU headroom (~7.9k of 18k CPU-s used), (b) build the ensemble with per-model feature subsets and
varying max_bin/leaves so members decorrelate by architecture rather than seed alone, (c) calibrate the
number of boosting rounds on a time-blocked split (late-2005 vs early-2005) to better mimic the
2005→2006 shift instead of eval-based early stopping, and (d) try quantile-binned dep-time and
per-airport hour profiles as features, which the current TE scheme never tested.
