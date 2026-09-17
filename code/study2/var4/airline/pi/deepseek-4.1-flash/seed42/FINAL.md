# FINAL — airline delay prediction (XGBoost)

**Best Eval AUC: 0.7425** (experiment #40, commit `8b82e6c`)
Baseline was 0.7141 (30 trees, raw features). Total gain: **+0.0284 AUC**.

## Final model
- Features: raw columns + `dep_hour = (DepTime//100)%24`, `dep_min`, `time15 = dep_hour*4 + dep_min//15`
  (15-minute categorical), and `carrier_hour = UniqueCarrier + "_" + dep_hour`.
- `dep_hour`, `carrier_hour`, and `time15` are passed as native XGBoost categoricals; all other
  object columns (Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest) are categorical with levels
  fixed from the training data.
- Model: ensemble of 8 `XGBClassifier` members, depths `[5,6,7,8,9,10,6,10]`, `n_estimators=800`,
  `learning_rate=0.03`, `tree_method="hist"`, probabilities averaged.
- All feature engineering and encoder fitting live inside `prepare()` / module-level train-derived
  constants, so `predict_proba(df)` reproduces the exact pipeline on the hidden holdout.

## Changes that mattered most
1. **`carrier_hour` categorical (+0.0143, exp18).** The single biggest jump (0.7196 → 0.7339). Airline
   scheduling banks give carrier-specific delay-vs-hour patterns that transfer across the 2005→2006 shift.
2. **`dep_hour` as a categorical (+0.0028, exp16).** Delay rate rises monotonically through the day and is
   very high overnight; letting the tree split hour groups explicitly beat treating it as numeric.
3. **Depth-diverse ensemble (+0.0032, exp33–34).** Averaging shallow-to-deep trees (depths 5–10) reduced
   variance substantially; seed averaging alone had given nothing.
4. **Finer time buckets (`time30` → `time15`, +0.0015 then +0.0013, exp39–40).** A 15-minute categorical
   time-of-day feature on top of `dep_hour` added consistent gains inside the ensemble.
5. **Model capacity (depth 8, 800–1200 trees, lr 0.03).** Depth 8 beat 6/7/10; lower LR + many trees
   helped slightly.

## Things that did NOT help
- **High-cardinality non-hour interactions / features**: `route` (Origin×Dest, 4198 levels, 385 unseen in
  eval), `origin_hour`, `carrier_dow`, `carrier_month`, `dow_hour`, `month_hour` — all overfit the 2005
  slice and lost 0.01–0.02 AUC.
- **Smoothed out-of-fold target encoding** of origin/dest/route/origin_hour/dest_hour: 0.7345 vs 0.7363.
- **Regularization / sampling**: `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`,
  `max_bin=1024`, `max_cat_threshold=256` all hurt, individually and inside the ensemble.
- **Frequency/popularity + is_weekend features**: 0.7356 vs 0.7363.
- **`dep_min` as categorical** (60 levels) badly overfit (0.7123).

## What I would try with more budget
The gains came almost entirely from *time-of-day structure* and *variance reduction*, so I would push
those further: (a) tune the ensemble composition more systematically (member depths, LR, `n_estimators`,
and possibly member-specific feature subsets) via a small held-out search rather than one-off experiments;
(b) explore multi-resolution time encodings (`time15`/`time30` plus smooth cyclic hour features) and
carrier×time interactions at different granularities, since 15-minute bins clearly beat 30-minute;
(c) test `lossguide`/`max_leaves` trees and DART as additional ensemble diversity; and (d) investigate
whether a within-train time-ordered validation split can select `n_estimators`/ensemble weights in a way
that tracks the year-shift better than the single 2006 slice. I would avoid any further high-cardinality
categorical interactions — they consistently overfit the 2005 training year.
