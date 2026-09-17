# Final report — airline departure-delay AUC

**Best Eval AUC: 0.7566** (experiment #12, commit `0639d7e`), up from the 0.7141 baseline.
Validated with `./validate.sh` → `CONTRACT OK` (predict_proba reproduces the full pipeline on raw rows).
Budget stopped the run at 17,764 / 18,000 CPU-seconds, so the loop was closed here.

## Setup
Task: predict `dep_delayed_15min` for US domestic flights. Train = 2005 (100k), eval = 2006 (100k),
hidden holdout = 2006-slice2. A key measurement: random CV **within 2005 reaches ~0.757**, while the
2005→2006 eval is ~0.72–0.76. The year boundary is the real difficulty — features that memorize
2005-specific calendar/weather patterns do not transfer.

## Changes that mattered most
1. **Dropping calendar features (Month, DayofMonth, day-of-year).** Month delay rates correlate only
   ~0.59 between 2005 and 2006, so trees fit 2005-specific seasonality. Removing them gave +0.007
   (0.735 → 0.742 in one step on the same model). DayOfWeek, in contrast, correlates 0.96 across years
   and was kept.
2. **Native XGBoost categoricals for UniqueCarrier / Origin / Dest.** Dropping them cost ~0.05 AUC.
   They transfer despite the year shift because airport/carrier effects are structural.
3. **Structural interactions, not target encodings.** `carrier × hour` (a categorical), origin-hour and
   destination-hour traffic-volume counts, and a diagonal `distance / (dep_minute+1)` ratio gave
   ~+0.010 combined. These encode congestion and delay-accumulation effects that hold across years.
4. **Deep, weakly-regularized trees with few boosting rounds** (`max_depth` 14–20, `lr=0.03`,
   ~200–250 trees, `colsample_bytree=0.4`, `min_child_weight≈1`). Much better than the baseline's
   shallow 30 trees; deeper trees exploit high-order categorical interactions.
5. **A 6-model XGBoost ensemble + `reg_alpha=1`.** Averaging six diverse configs (depth 14–20 plus a
   `lossguide`/512-leaf tree) reduced run-to-run variance (which is ~0.001 on eval) and added ~+0.003.

## What did not help
* **Target encodings** (origin/dest/carrier/route/hour), even with 5-fold out-of-fold values and
  smoothing: eval dropped to ~0.69. The 2005→2006 shift destroys the encoded propensities and lets the
  model over-rely on a leaky feature.
* **Origin–Dest route features** (route as a categorical, or route frequency): -0.01; too sparse and
  year-unstable.
* **Network/volume aggregates** (origin degree, carrier degree, total airport volume) and
  week-hour / origin-weekday counts: all neutral-to-negative.
* **Fitting more boosting rounds** (`n_estimators` 450–800) and stronger regularization
  (`reg_alpha≥2`, `reg_lambda≥3`, `max_bin≤128`, `gamma`, `subsample 0.7`): each lost 0.001–0.002.

## What I would try with more budget
Use the remaining wall clock to build a proper **time-aware validation** instead of selecting on the
single eval slice: e.g. carve an internal 2005 train/validation split that mimics the year shift (or
weight training rows to emphasize late-2005), so that keep/discard decisions are less exposed to eval
overfitting. Then explore **distribution-shift-robust modeling**: feature selection via importance
stability across resampled 2005 subsets, monotonic constraints on departure time, and a larger
seed-bagged ensemble (bagging was still reducing variance at the point the CPU budget ran out).
A second promising direction is richer *structural* interactions — e.g. carrier×origin×hour congestion
and connecting-airport flow — that capture operational delay propagation without relying on
year-specific target statistics.
