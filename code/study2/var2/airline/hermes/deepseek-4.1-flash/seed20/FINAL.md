# FINAL — airline (dep_delayed_15min) XGBoost autoresearch

**Best Eval AUC: 0.7483** (experiment #39, commit 1c2207d; baseline was 0.7141).
40/40 experiments used, ~36 min of the 230 min wall clock, 5.9k of 18k CPU-seconds.
Final `train.py`: multi-model XGBoost ensemble, all feature engineering inside `prepare()` (validated: `CONTRACT OK`,
eval AUC 0.7483 when the target column is removed and `predict_proba(df)` is used).

## Changes that mattered most

1. **Time/calendar feature engineering inside `prepare()`** (+0.0035 over baseline). `DepTime` split into
   Hour / Minute / HourFrac plus cyclic hour, numeric Month / DayofMonth / DayOfWeek plus their sin/cos, log
   Distance. The raw `DepTime` integer hid the time-of-day structure the trees needed.
2. **Capacity with leaf-wise growth** (+0.004). 30 trees/depth 6 → 200-300 trees; `grow_policy="lossguide"` with
   `max_leaves` 512-2048 clearly beat `depthwise` depth-10/12. `min_child_weight` must stay low (5): raising it to
   20 cost 0.004.
3. **Smoothed out-of-fold target encodings** (+0.001). Carrier, Origin, Dest, Origin×Hour, Carrier×Hour,
   Carrier×Origin, Carrier×Dest, with 5-fold OOF values for the training rows and full-train maps for unseen rows
   (`m=20` for the low-cardinality keys, `m=50` for crosses).
4. **Stable traffic-volume features** (+0.002). Train-derived counts for Carrier / Origin / Dest / Origin×Hour /
   Route plus the shares RouteCount/OriginCount and OriginHourCount/OriginCount. These are structural rather than
   identity features, so they survive the 2005→2006 shift.
5. **Heterogeneous 7-model ensemble + mild recency weighting** (+0.008 total). Seven XGBoost configs (lossguide
   512/1024/2048 leaves, depthwise depth 8/12, two numeric-only members) averaged, each trained with sample
   weights `1 + (month-1)/11` so later 2005 months count more. `max_cat_threshold=256` gave a further +0.001.

## What did not help

- **Route/airport-pair identity features.** `Route` as a categorical (-0.008 at fixed hyperparameters) and
  `te_route` (-0.0007) both hurt: the pair-level signal is year-specific. Route *volume* helps; route identity does not.
- **Plain frequency encodings of identity columns** used alone (CarrierCount/OriginCount/DestCount at
  30 trees, -0.005): raw 2005 counts drift with traffic growth. Only after the model was strong enough and the
  counts were joined with peak-hour/route context did volume become useful.
- **Aggressive regularization and extra trees**: `min_child_weight=20` (-0.004), 400 trees at depth 10 (-0.001),
  depth 12 (-0.0002), exponential recency weights 0.85^(12-m) (-0.0017), row-bagging members at 85% (-0.0011),
  an 8th shallow member (-0.0009), a DART member (-0.0001). Also extra calendar features (DayOfYear, IsWeekend)
  cost 0.002, and 10-fold instead of 5-fold OOF target encoding changed nothing.

## With more budget

The remaining headroom looks like it is in *shape* rather than in more features. I would fit a per-model
weighting scheme (e.g. non-negative least squares or a small logistic stacker on an internal 2005 holdout
carved out by time, not randomly) instead of the plain average, since the members differ a lot in strength;
run a proper time-series CV (train on month *m*…*m+k*, validate on the next month) to choose `max_leaves`,
`lr` and the recency half-life, which I could only probe with single eval-guided points here; and add
peer-group aggregates computed from train (per-carrier average distance, per-origin share of delayed-prone
banks) plus day-of-week × hour target encodings restricted to high-count cells, which the 5-fold scheme
should keep honest. Two directions I deliberately did **not** take: training on `data/eval.csv` (it is 2006 data
and would raise the reported AUC while destroying the only honest generalization signal I have) and embedding
the eval year's drift by feature-engineering `eval.csv` statistics into `prepare()`.

Runtime note: the final `train.py` needs ~110 s of the 120 s experiment budget; a cheaper 7-member variant
(#40, member 3 at 200 trees) scored 0.7482 in 103 s if wall-clock headroom is ever preferred over the 0.0001.
