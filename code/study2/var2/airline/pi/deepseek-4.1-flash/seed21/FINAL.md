# Final Report — airline delay (XGBoost), best Eval AUC: **0.7603**

Task: predict `dep_delayed_15min` (Y/N). Train = 2005 slice (100k), eval = 2006 slice1 (100k),
hidden holdout = 2006 slice2 (1m). Metric: AUC. Budget: 40/40 experiments used, ~8500/18000 CPU-s.

Final model: an ensemble of five XGBoost classifiers whose predicted probabilities are averaged.
Members: one `grow_policy=lossguide, max_leaves=16384` model plus four depthwise models
(`max_depth` 20–24, `colsample_bytree=0.4–0.5`, `max_bin` 512/1024, `lr=0.03`). Individual model eval
AUCs are 0.7577–0.7586; the average reaches 0.7603.

## Changes that mattered most
1. **Structural / count features fitted on train only** (biggest feature gain, +0.006): flight counts per
   Origin, Dest, UniqueCarrier, route `Origin_Dest`, and Origin×hour; carrier network size (distinct
   routes); route competition (distinct carriers per route); Origin/Dest connectivity (distinct
   destinations/origins); mean distance per Origin and per Carrier. These are traffic-structure features
   that stay valid across the 2005→2006 shift.
2. **Much higher tree capacity** (biggest single lever, ~0.720 → ~0.756): the baseline's 30 shallow trees
   were badly underfit. Deep trees (`max_depth` 8→24), or unbounded loss-guided trees with many leaves,
   capture high-order interactions among hour/distance/airport-size/carrier that persist across years.
3. **Low `colsample_bytree` (0.4–0.5) with deep trees**: keeps individual splits weak while allowing the
   many trees needed at high depth; clearly better than 0.8.
4. **`max_bin` 512/1024**: finer split points gave a small but consistent gain (0.7583 → 0.7586) and add
   useful ensemble diversity.
5. **Diverse ensembling** (averaging probabilities of loss-guide + depthwise models across seeds):
   +0.0013 over the best single model and much more stable than any one run.

Also kept: hour-of-day and cyclical time-of-day (sin/cos of `DepTime`), carrier as the only native
categorical (low cardinality).

## What did NOT help (and why it matters)
1. **Target encoding** for carrier/origin/dest/route, even with out-of-fold fitting and strong smoothing
   — all variants lost 0.0005–0.001. Label statistics from 2005 do not transfer to 2006.
2. **Route / calendar categoricals and smooth seasonality** (`Month`, `DayofMonth`, `DayOfWeek`,
   month/day-of-year sin-cos): catastrophic (0.737–0.749 vs 0.758). The model over-relies on year-specific
   seasonality that shifts between years. `min_child_weight`, `reg_alpha=5`, `reg_lambda=5`, log-distance,
   and additional count features (carrier-route, origin-month, carrier-hour) also gave nothing.
3. **Shallow models / early stopping and small tree counts**: internal random-split validation kept
   improving up to 500+ trees while eval peaked much earlier, so early stopping selects an underfit
   model — the year gap, not the data size, is the bottleneck.

## With more budget I would try
A more principled validation scheme that simulates the year shift (e.g., train on 2005 months 1–6 and
validate on 2005 months 7–12) to tune capacity without overfitting eval.csv; a wider randomized
hyperparameter search over `max_leaves`/`max_depth`/`max_bin`/`colsample` with ensembling of the top
configs; explicit interaction features (departure-hour × airport-traffic, distance × carrier) to reduce
reliance on very deep trees; and rank-averaging or weighted ensembling (weights fitted on held-out 2005
months) instead of the plain probability mean. All feature statistics must remain fit on training data
only, since `predict_proba` reapplies them to the hidden holdout.
