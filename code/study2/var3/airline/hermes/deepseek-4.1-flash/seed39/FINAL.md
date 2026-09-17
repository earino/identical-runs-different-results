# FINAL — airline delay prediction (XGBoost, 40 experiments)

**Best Eval AUC: 0.7479** (commit `3600251`, experiment #39). Baseline was 0.7141 (+0.0338).
Final `train.py` runs in ~64 s (limit 120 s), contract validated: `./validate.sh` → `CONTRACT OK`,
and `predict_proba(eval.drop(columns=[target]))` reproduces 0.7479 with the target column absent.

## Changes that mattered most

1. **Departure-time representation (exps 2, 9, 11, 12, 20, +0.0199).** `DepTime` (hhmm) → minutes since
   midnight, cyclic sin/cos, *and* multi-scale **native categorical buckets** (24-hour, 30-min, 15-min).
   XGBoost's categorical partition splits pick optimal time-window groupings in a single split, which is
   worth much more than numeric splits on a smooth time feature (hour categorical alone: +0.0060).
   `max_cat_threshold` raised 64 → 128 so the 96-level bucket is not truncated.
2. **Estimated arrival time (exps 33, 38, +0.0041).** Block time ≈ 30 min + `Distance`/480 mph, then
   `arr = (tod + dur) mod 1440`, with its own 15- and 30-min categorical buckets. This is an explicit
   time-of-day × distance interaction that trees otherwise have to build across several levels.
3. **Bagged XGBoost ensemble (exps 15–19, 24–26, 36, +0.0040 from bagging itself).** 9 members with
   different seeds, row/column subsampling, depth (6/7/8), `reg_lambda`, learning rate and tree count,
   averaged. Reliability gain over any single model.
4. **Leaf-wise (`grow_policy="lossguide"`) members with a leaf budget** (255 leaves, depth ≤ 12) — more
   flexible time-of-day partitions than depth-wise trees. Kept bounded on purpose: the unbounded variant
   scored 0.7433 but took exactly 120 s (one run away from a timeout kill).
5. **Small robustness knobs:** `max_bin=512` (+0.0002), `min_child_weight=3` (+0.0001).

## What did not help (reverted)

1. **High-cardinality categoricals.** Route `Origin_Dest` (4198 levels) cost **−0.014**; `month×dayofmonth`
   (366 levels) cost **−0.033**; 5-min (288) and 10-min (144) time buckets also lost. Cardinality around
   24–96 is the sweet spot; above that, partition splits fit per-level noise.
2. **Target/count encodings** with out-of-fold fitting for Origin/Dest/Carrier: 0.7196 vs 0.7199 — neutral,
   ~40 lines of machinery for nothing.
3. **Capacity/regularization tinkering:** 2000 trees at lr 0.03 (−0.0019), depth 5 + `min_child_weight` 10 +
   `reg_lambda` 2 (−0.0022), per-member early stopping on a 10% internal split (−0.0017), subsample 0.6
   (−0.0002). Also marginal and reverted: holiday-period flag (+0.0003), `max_cat_to_onehot=16` (+0.0000),
   arrival sin/cos (−0.0000 but 2 fewer features, so dropped for simplicity).

## What I would try with more budget

The last ~0.005 of measured gains are within eval noise (AUC SE ≈ 0.002 at n = 100k), so the next step is
better *selection*, not more feature archaeology: build several bags and choose on a year-shifted internal
split (fit on part of 2005, validate on the rest) rather than on `eval.csv`, which should separate real
generalization from eval-fitting. On the modelling side, the one clearly unexploited signal family is
aircraft rotation/bank congestion: no tail number or prior-leg time exists here, but train-only statistics
such as departures per route-hour or per origin-hour (frequency and smoothed delay propensity, fitted on
2005 with strong shrinkage) are a plausible proxy, and the `tod` × `Origin` interaction is exactly where
the current trees are weakest. Beyond that, a proper search over the leaf-wise family (leaves × lr ×
subsample) with a wall-clock budget, and weights over bag members chosen by internal CV instead of equal
averaging, are the obvious remaining levers.
