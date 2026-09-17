# FINAL — airline departure-delay AUC (autoresearch, 40 experiments / 230 min budget)

**Best Eval AUC: 0.7296** (experiment #40, commit `64685f4`, `train.py` at HEAD).
Baseline was 0.7141. Total gain: +0.0155 AUC. 40/40 experiments used, ~37 s per run, well inside the
120 s / 6 GB / thread limits.

## What the final model is

* Features: `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest` (XGBoost categoricals, levels fit on train),
  `Distance`, `DepTime`, `dep_hour`, `dep_min`, plus train-only aggregate features
  (`cnt_origin`, `cnt_dest`, `cnt_route`, `cnt_origin_hour`, `cnt_dest_hour` — all log1p),
  `dep_frac` (fraction of the origin airport's daily departures already gone),
  and destination-side congestion at the estimated arrival hour
  (`arr_hour`, `cnt_dest_arrhour`, `cnt_dest_dep_at_arr`, `dest_arr_frac`).
* Model: bag of 16 XGBoost models — 8 depth-wise (`max_depth` 2–6, 250–800 trees, lr 0.03–0.06,
  `subsample` 0.5–0.9, `colsample_bytree` 0.4–1.0) and 8 leaf-wise (`grow_policy="lossguide"`,
  `max_leaves` 12–96). Predictions are the plain mean of member probabilities.
* All statistics/encoders are computed from `data/train.csv` only, inside module-level state that
  `prepare(df)` reads — so `predict_proba` reproduces every transform on unseen rows.
  `./validate.sh` prints `CONTRACT OK` (eval AUC via `predict_proba` with the target column removed: 0.7296).

## The changes that mattered most

1. **Dropping year-unstable features (+0.0037).** With train=2005 and eval/holdout=2006, per-group delay
   rates do not transfer equally: `DayofMonth` corr 0.33, `Month` corr 0.59, but `DayOfWeek` 0.96,
   `UniqueCarrier` 0.85, `dep_hour` 0.96. Removing DayofMonth took 0.7174 -> 0.7177, removing Month
   0.7177 -> 0.7211. Screen candidate features by cross-year correlation before trusting them.
2. **Schedule/congestion aggregates fit on train only (+0.0022).** Airport *volume* transfers almost
   perfectly across years (corr 0.97), unlike airport *delay rates* (corr 0.38). log1p counts of
   origin / dest / route / origin×hour / dest×hour took 0.7232 -> 0.7254.
3. **Explicit time-of-day + day-position features (+0.0042).** `dep_hour`/`dep_min` (0.7211 -> 0.7232),
   then `dep_frac` — rank of this flight's `DepTime` inside the origin's daily departure distribution —
   and the destination-side mirrors (`arr_hour`, `cnt_dest_arrhour`, `cnt_dest_dep_at_arr`,
   `dest_arr_frac`) — 0.7254 -> 0.7296. Congestion at both ends of the flight, normalised to each
   airport's own schedule, is where the remaining signal was.
4. **Shallow trees + subsampling instead of raw capacity (+0.0019).** 30 trees / lr 0.1 / depth 6 = 0.7141;
   depth 4, 400 trees, lr 0.05, subsample/colsample 0.8 = 0.7160. Depth 6 with 500 trees, `min_child_weight`
   20 and `reg_lambda` 5 still scored only 0.7100 — on this task complexity buys 2005-specific noise.
5. **Heterogeneous bagging (+0.0011, and it made item 3 pay off).** 5 seeds of one config: 0.7160 -> 0.7170;
   8 mixed-depth members: 0.7174; adding 8 `lossguide` members to the 8 depth-wise ones: 0.7257 -> 0.7283.
   Averaging decorrelated tree shapes is the only "more compute" knob that never hurt.

## What did not help (all reverted)

* **More capacity on the raw feature set**: 400 trees lr 0.05 (0.7115), early stopping on an internal 20 %
  split that picked 333 trees (0.7098), depth 6 + `min_child_weight` 20 + `reg_lambda` 5 (0.7100).
* **High-cardinality identity encodings**: routing `Origin_Dest` as one more categorical hurt early on
  (0.7089); `origin_hour`/`dest_hour` categoricals scored 0.7186; `max_cat_threshold=8` 0.7234 (the wide
  default partition splits are genuinely useful); out-of-fold target encoding of route/origin/dest/carrier
  was exactly neutral (0.7160) on top of categoricals.
* **Re-adding coarse seasonality / extra counts / padding the bag**: peak-season flag from Month 0.7253,
  `cnt_route_hour` + `cnt_carrier_route` 0.7239, `cnt_route_rev` + hub flags 0.7257, 20 members with
  `max_bin` variants 0.7281, 20 members padded with depth-2/6 and `colsample_bynode` members 0.7278,
  12 depth-wise + 8 lossguide 0.7283. `booster="dart"` members timed out past 120 s.
* Also neutral-to-negative: dropping `Origin`/`Dest` entirely (0.7056 — identity matters, so the fix was
  to add stable aggregates, not to remove airports), `dep_minutes` (0.7256), TE on top of counts.

## With more budget I would

Do stability screening first, as a routine rather than as a mid-run discovery: for every candidate
feature, measure how well its per-group delay rate correlates between the two years and keep only the
part that carries over (this is what produced the two biggest jumps here). Then build a proper
stacking layer — per-carrier or per-hub sub-models plus a second-level XGBoost on out-of-fold
predictions, with member weights fitted on an internal time-based split instead of a plain average,
since equal weighting of 16 members is certainly leaving a little on the table. For features I would
push on delay *propagation* rather than delay *description*: aircraft-rotation proxies built from
repeated (carrier, route, hour) blocks, a holiday/peak calendar instead of the coarse month flag I
tried, and time-zone-normalised local hour for the destination. For the model I would re-do the
bag with a cheap dropout approximation (the tree booster's `rate_drop` rather than `booster="dart"`,
which I never got to finish inside the 120 s cap) and run a coarse random search over depth /
`max_leaves` / subsample with the internal validation split. Finally, since eval is only 100 k rows
and the last five accepted changes were each worth 0.0002–0.0006, I would replicate the final few
comparisons on a second holdout slice before believing them.
