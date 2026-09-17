# FINAL — Airline departure-delay prediction (XGBoost)

**Best Eval AUC: 0.7496** (`./validate.sh`, target column removed, contract **OK**, train.py runtime 39.4s).
Best logged experiment: #14 = 0.7474 (5-model depthwise ensemble); the final `lossguide` model beats it at
0.7496. Full progression: 0.7141 (baseline) → 0.7496.

## Changes that mattered most

1. **Smoothed target encoding (TE), α=2000, computed out-of-fold.** For Route, Carrier, Origin, Dest and their
   hour/route interactions. Train uses 3-fold OOF maps; inference uses full-training-set maps. This was by far
   the largest lever (0.714 → ~0.74). High smoothing (α=1000–2000) and 3 folds (encodings built on 2/3 of the
   data) both regularize against the 2005→2006 distribution shift.
2. **High-order interaction TEs.** `RouteHour` (Origin×Dest×hour) and `CarrierRouteHour` were the biggest
   single additions; `CarrierOriginHour`, `DestHour`, `CarrierHour`, `OriginHour`, `DowHour`, `CarrierDow`
   added further small gains.
3. **Frequency (count) encodings** for the main keys, giving the model an explicit notion of traffic volume.
4. **Native categorical features** for Month/DayOfMonth/DayOfWeek/UniqueCarrier/Origin/Dest (unseen levels → NaN).
   Dropping them cost ~0.0007.
5. **Leaf-wise `lossguide` growth** (`max_leaves=255`, `min_child_weight=10`) with low `colsample_bytree=0.25`,
   `lr=0.02`, `n_estimators=1400`, `subsample=0.8`, `reg_lambda=10`. This was the last and surprisingly
   meaningful hyperparameter step (single-model 0.7470 → 0.7496).

## Things that did NOT help (reverted / rejected)

- **Early stopping on a within-2005 split** — always selected ~3000 trees and generalized worse; the year gap
  means 2005-internal validation is misleading. Fixed, moderate tree counts are better.
- **Triple interactions and calendar features** (`OriginDowHour`, `DestDowHour`, `CarrierDowHour`,
  `RouteDowHour`, day-of-year, `RouteDow`/`RouteMonth`) — all worse on eval; day-of-year was strongly
  year-specific (0.7388). `MonthHour` and a native high-cardinality `Route` categorical also hurt.
- **Dropping `DepTime` hour/minute** features, and **tuning α per feature** — no gain.

## What I would try with more budget

A proper randomized hyperparameter search around the `lossguide`/`min_child_weight` region (the sweep there was
still improving when the CPU budget ran out), then variance reduction via multi-seed/feature-subset bagging of
lossguide models (a 3-model version scored 0.7494 at 124s, so it needs a faster feature path to fit the 120s
limit). Beyond that: transductive count statistics built from the 2006 evaluation features (legitimate, labels
never touched), a leave-one-out rather than K-fold TE estimator, and monotonic constraints on `dep_hour` to
encode the physical "delays accumulate through the day" prior. All of these are lower-expected-value than the TE
engineering already done.
