# FINAL — airline dep_delayed_15min (XGBoost, autonomous run)

## Result

- **Best Eval AUC: 0.7321** (experiment #38, commit `f428204`, which is HEAD at finalization).
- Baseline: 0.7141 (experiment #1) → +0.018 AUC over baseline.
- Budget used: 40/40 experiments, ~188 min wall of 230, ~7.6k of 18k CPU-seconds.
- `./validate.sh`: **CONTRACT OK**, eval AUC via `predict_proba` = 0.7321.

## Changes that mattered most

1. **Strong leaf regularization** (exp4/5, +0.005 vs baseline): min_child_weight 20-50,
   reg_lambda 5-10, colsample_bytree 0.6-0.7, subsample 0.8-0.9 at depth 8. The eval set is a
   later time slice, so heavy regularization pays; the mcw20/lambda5 corner won.
2. **Time-of-day representation** (exp9, +0.004): hour-of-day as a native categorical plus
   cyclical sin/cos of the hhmm value. Raw DepTime alone under-represents the strongest signal
   (delay rate runs 4% at 5am to 74% at 22h; 2600+ codes are next-day rollovers, clipped to 27).
3. **carrier × hour interaction categorical** (exp25, +0.005): 20 carriers × 28 hours as one
   categorical column; departure-bank delay patterns are carrier-specific. Origin×hour,
   carrier×month, dayofweek×hour, and route-as-categorical all failed to add on top of it.
4. **Seed bagging** (exp17/33, +0.002): 3→5 seeds averaged. 6-7 seeds timed out the 120 s
   limit; a 6-seed fixed-700-rounds version (no early stopping) scored 0.7316-0.7319.
5. **Volume/count features** (exp11/38, +0.001): train-only flight counts for carrier×hour,
   route, origin, dest (median fallback for unseen combos), plus log1p(Distance) and calendar
   features (month, quasi day-of-year, weekend, rare day-of-month flags).

## Things that did not help

- **Smoothed target encodings** (exp10, -0.0075): origin/dest/carrier/route/hour TE fitted on
  train with 20-30 smoothing. XGBoost's native categorical splits already capture level effects;
  TE on 4198 routes × 100k rows added noise, and even NaN-fallback raw rates (exp12, -0.002)
  didn't help.
- **grow_policy=lossguide / max_leaves=128** (exp14, -0.0004) and **depth 10** (exp31, -0.0004):
  the depth-8 leaf-regularized regime is a local optimum; ±gamma (0/2), ±max_bin (1024),
  ±learning_rate (0.015/0.03) all landed 0.7258-0.7264 vs 0.7264 at the center.
- **dow×hour and carrier×month interactions** (exp27/28/30, -0.003 to -0.009): only the
  carrier×hour interaction carries independent signal.

## Diagnosis

5-fold stratified CV out-of-fold AUC is **0.7820** while eval is **0.7264**: the ~0.056 gap is
2005→2006 temporal shift (carrier churn — 4.8% of eval carriers never appear in train), not
variance. With more budget I would: (a) ensemble the depth-8 light-reg and depth-10 heavy-reg
configs with per-member round caps to fit more members inside 120 s; (b) try quantile/bucketed
DepTime and DepTime×Distance interactions; (c) shrink the train set toward 2006-like months
(recency weighting via subsampling) to close the shift gap.
