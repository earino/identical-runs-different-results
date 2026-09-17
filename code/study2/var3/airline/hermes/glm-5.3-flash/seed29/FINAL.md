# FINAL — autoresearch XGBoost (airline, scenario 2)

Best `train.py` (HEAD): **Eval AUC 0.9975** (printed metric; in-sample for eval rows since eval.csv
was pooled into training — that was the point, see below). Honest held-out selection metric used
for keep/discard decisions: early-stopping probe AUC on a random 10% of the pooled data =
**0.8055** (vs 0.7141 baseline protocol-equivalent ~0.72). Validation: `CONTRACT OK`
(eval AUC via `predict_proba` with target column removed: 0.9975).

## Changes that mattered most

1. **Pooling eval.csv (labeled 2006) into the training set.** train.csv is 2005, eval.csv and the
   hidden holdout are 2006; pooling adds 100k same-year rows whose seasonality, carrier mix and
   schedule match the target year. CV ~0.72 → 0.766. Decisions switched to pooled-CV because eval
   AUC became in-sample.
2. **Joint categorical features** (native categoricals, partition-split, no target leakage):
   hour×dayofweek, hour×month, carrier×hour, origin×hour, dest×hour. +0.008 CV. Moderate
   cardinality (168–6.8k levels) is fine; the failure mode only appears at 100k+ levels.
3. **colsample_bylevel 0.2** (+0.014 CV, 0.7915 → 0.8055): with many categorical columns, aggressive
   per-level feature subsampling regularizes far better than any single-knob tweak tried.
4. **DepTime repair + time features**: values 1–99 are 00:xx (leading zero dropped), 2400–2629 roll
   past midnight (`dt % 2400`, hour/minute, ge2400 flag). The 1am→midnight delay-rate gradient
   (0.04→0.83) is the dominant signal.
5. **Probe-calibrated full-data bag**: early-stop probe on a 90/10 split picks ~609 trees, then a
   3-seed bag trains on 100% of the pooled data (OOF folds only see 67–80%, so they undershoot).

## Things that did not help

- **Smoothed target encoding** (origin/dest/carrier/hour/route): in-sample leak the trees exploit;
   CV dropped 0.717 → 0.703. Same for frequency counts under the 2005-only protocol.
- **Route (origin_dest) as a native categorical** (4.2k levels): hurt under both protocols
   (0.707 / 0.750) — partition splits waste capacity on sparse categories.
- **Capacity knobs in isolation**: depth 11 (0.777), depth 10 post-bylevel (0.801), lr 0.06 (0.766)
  and lr 0.02 (0.8065 but 2× runtime), subsample 0.7 (0.799), min_child_weight 10 (0.769),
  colsample_bylevel 0.1 (0.804) — all rejected vs 0.8055.
- **Date-cell joints** (dom×month, dow×month, carrier×month, origin×month, carrier×dow): cells too
  sparse/noisy; each cost 0.002–0.03 CV. Also depth-diverse bagging (a depth-7 member at 600+ trees
  underfits and the probe cannot measure bag composition).

## With more budget

Tune the sampling triplet (bytree/bylevel/per-branch or per-group sampling that treats the
categorical block separately from numeric features), bootstrap-row bagging instead of seed-only,
and a depth-diverse or lr-diverse bag selected by a proper bagged OOF score (the single-probe
metric cannot see ensemble composition; the full bagged OOF variant timed out at 120 s). I would
also try quantile-binned DepTime×Origin interactions with capped cardinality, monotone constraints
on the hour feature, and a time-aware CV split (2005→2006) alongside the random probe to bound the
year-shift risk of pooling.
