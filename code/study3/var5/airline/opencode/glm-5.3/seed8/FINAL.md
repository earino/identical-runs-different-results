# Final Report — autoresearch XGBoost (airline delay)

## Result

- **Best Eval AUC: 0.7569** (experiment #11, commit `98379c1`), up from the 0.7141 baseline (+0.043).
- Final config: 7-member config-jittered XGBoost ensemble, probability-averaged; trains in ~110–119 s, passes `validate.sh` (CONTRACT OK).

## Changes that mattered most

1. **Dropping Month / DayofMonth entirely** (keeping DayOfWeek). With train=2005 and eval=2006, any calendar
   partition lets deep trees memorize year-specific delay patterns. This one insight was worth ~+0.01 and
   everything downstream was built on it.
2. **One-hot encoding (not native categorical) + deep trees**: one-hot for DayOfWeek/UniqueCarrier/Origin/Dest
   (levels fit on train only) + hour-of-day one-hot, then `max_depth=24, n_estimators=200, lr=0.05,
   subsample=0.8, colsample_bytree=0.6`. One-hots made depth essential (d16→d24 = +0.0045).
3. **Stable numeric features, all fit on train only**: `te_qh` (smoothed target encoding of quarter-hour
   departure bins, m=100), `org_cnt`/`dst_cnt`/`route_cnt` (log1p flight volumes of Origin, Dest, and
   Origin→Dest pair; dst +0.0017, route +0.0009 — volumes transfer across years even where identities don't),
   `log_dist`, `dist_vs_route`, `minute`.
4. **Config-diverse seed bagging**: 7 members, seeds 0–6, jittered over (colsample_bytree × max_bin)
   = {(0.60,256),(0.55,512),(0.65,256),(0.60,512),(0.55,256),(0.65,512),(0.60,256)}; averaged predictions.
   Bagging 1→7 ≈ +0.002; config jitter decorrelates members beyond seeds alone.

## Things that did not help (all reverted)

1. **Route identity in every form**: native categorical route (0.69), top-300 route one-hot (0.7510), route
   target-encoding (0.7418). Pair *volume* helps; pair *identity* is pure 2005 memorization (385 unseen eval routes).
2. **Any calendar-derived feature**: month one-hot or cyclic (−0.01), season bins, even recurring US-holiday
   proximity dates (0.7429). Calendar is poison under a year shift, full stop.
3. **Everything else tried**: carrier-volume counts (−0.004), all carrier/airport/dow×hour target encodings,
   CV row-bagging (0.7482), lossguide growth (0.7483), early stopping (train-only fit won), >200 trees,
   min_child_weight/lambda/gamma, DepTime jittering for member diversity, ordinal-view ensembling,
   distance-bin one-hots, lr jitter, te_qh smoothing m=300.

## With more budget

I would raise ensemble member count with wider config jitter (the 120 s experiment cap forced 7 members and a
119 s final runtime — the single biggest remaining lever is averaging 15–30 decorrelated members). I would also
build an internal *time-blocked* validation inside 2005 (train Jan–Sep, validate Oct–Dec) to screen features by
shift-robustness instead of a single 2006 eval slice, then mine the volume-feature family more systematically
(per Origin×hour congestion counts, rolling window statistics), since volume features were the only late-stage
wins (dst_cnt, route_cnt) and their generalization mechanism is clear.
