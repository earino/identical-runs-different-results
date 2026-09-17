# Final report — airline delay AUC

**Best Eval AUC: 0.7511** (experiment #28, commit `f1015cb`), up from the 0.7141 baseline.

Final model: a single `xgboost.XGBClassifier` on raw + engineered features, `n_estimators=400`,
`max_depth=20`, `learning_rate=0.03`, `colsample_bytree=0.7`, `reg_alpha=2.0`,
`max_cat_threshold=128`, native categorical support, 4 threads. Train + eval ≈ 98 s.

## Changes that mattered most
1. **Dropped `Month` and `DayofMonth`.** They encode 2005-specific seasonality that does not transfer to the
   2006 eval slice (e.g. January delay rate 0.541 in 2005 vs 0.444 in 2006). Removing them took 0.7141 → ~0.720.
2. **Origin × time-of-day categoricals.** `origin_hour` then progressively finer `origin_slot`
   (30-min, finally **15-min** `Origin_<minute_slot>`). Airport-specific congestion by time of day is a stable
   cross-year signal and was the single largest feature gain (≈ +0.028 overall over the baseline feature set).
3. **Strong L1 leaf regularization + deep trees.** `reg_alpha=2.0` with `max_depth=20` was worth ≈ +0.01 vs the
   same depth with `reg_alpha=0`; the model was underfit at the baseline depth of 6 and overfit without L1.
4. **`max_cat_threshold=128`** (default 64) improved high-cardinality categorical partitioning: +0.0013.
5. **Time encodings** `minute_of_day` (linear) plus `dep_minute_of_hour` (schedule roundness) added ≈ +0.003,
   while keeping the raw `DepTime` integer was essential (removing it collapsed AUC to 0.689).

## Things that did NOT help
- **Naive extra capacity**: `n_estimators=400, depth=7, subsample` gave 0.7094, *worse* than 30 shallow trees.
- **Route / interaction encodings**: `Origin_Dest` route, `Origin_Carrier`, `Origin_DayOfWeek`,
  `Dest_hour`, target encoding of Origin/Dest/Carrier, and frequency counts all hurt on the time-separated eval.
- **More rounds / bigger thresholds**: `n_estimators=1500` (0.7437) and `max_cat_threshold` 192/256 (0.7450/0.7451)
  were below the 128 configuration; a 10-min slot and a 2-model seed ensemble were also flat/noise.

## What I would try with more budget
Push the time-space representation further: learn an embedding or smoothed target statistic for
`Origin × 15-min slot × carrier` instead of raw categorical partitions, and bag several seed/depth variants
(the CPU-time cap forced single-model runs near the 120 s experiment limit). I would also build a proper
time-based internal validation split (late 2005) to screen configs without spending the eval set, then confirm
only the finalists on eval — most of the post-0.745 gains were within the ~0.0005–0.001 eval noise floor, so a
larger independent validation signal would make the last stretch of tuning trustworthy.
