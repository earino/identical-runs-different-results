# Final Report — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7635** (experiment #40, commit `c31a031`; baseline was 0.7141, +0.049 total).
Budget used: 40/40 experiments, ~159 min of 230, ~11,000 of 18,000 CPU-seconds.

## Final architecture

`prepare()` builds 40 features per row (all computed identically for train/eval/hidden holdout):

- Congestion counts (log1p of same-day schedule sizes, computed within the dataframe passed in):
  per (day, origin), (day, dest), (day, origin, hour-block), (day, dest, hour-block), (day, carrier),
  (day, origin, carrier), (day, hour-block, carrier), (day, system), (day, route).
- OOF target encodings (5-fold, smoothed) on hour-chain keys: carrier/origin/dest × hour,
  route, hour, dep-time block, origin × hour-block, hour × cong-day-bucket.
- Cyclical time-of-day (sin/cos), day-of-year seasonality, day-of-week, Distance, is_night.
- Volume counts (log) per carrier/origin/dest/route.

Model: mean of 12 diverse XGBoost bags (depth 6–9, colsample 0.4–0.9, mcw 20–80), each trained on
90% of train with early stopping on its own 10% holdout (round cap 2400). Categoricals via
`enable_categorical`; raw string categoricals for Month/DayofMonth/Dow are NOT used (they hurt).

## Changes that mattered most

1. **Day-level congestion features** (exp13/14/15/24, 0.718→0.762): log1p flight counts per
   (day×origin), (day×dest), (day×carrier), and especially (day×system total) — corr 0.20 with the
   target, perfectly stable across the 2005→2006 years. The single biggest gain (+0.021 in one step).
2. **Bagged ensemble with per-bag early stopping** (exp8, 0.720→0.727): each bag early-stops on its own
   random 10%, which both regularizes and decorrelates the bags; +0.0075, more than any single feature.
3. **OOF target encodings on hour-chained keys** (exp3→7, 0.715→0.720): hour TE alone has eval AUC
   0.69 (the delay rate climbs monotonically 0.04→0.98 through the day); origin×hour, dest×hour,
   carrier×hour, origin×hour-block add complementary interaction structure.
4. **Carrier-level congestion** (exp24, 0.7633): same-day flights per carrier and per (origin, carrier).
5. **Ensemble sizing with config diversity** (exp10/29/40): 12 bags with varied depth/colsample beat
   6–8 bags by a hair; probability-averaging stayed ahead of rank-averaging and stacking.

## Things that did NOT help

1. **Full-train target encodings for training rows** (A/B test): eval 0.7437 vs 0.7627 for OOF encodings
   — leakage into the fit dominated the higher internal validation AUC.
2. **Date-specific target encodings** (month×day-of-month: 0.7621 vs 0.7626; month×hour, dow×hour all
   slightly negative): month effects flip sign between 2005 and 2006 — anything year-localized overfits.
3. **DART boosters, rank:pairwise objective, monotone constraints, deeper/shallower trees, losguide**:
   all within noise of the logistic/hist baseline; ranking and DART slightly worse.

## With more budget

- Optimize the cong-day smoothing/round cap jointly on a nested CV inside train (the 115s runs sit
  right at the 120s limit; a faster feature pipeline would let me push to 20+ bags).
- Grow the OOF-TE key list with per-key smoothing levels tuned by OOF AUC (HOUR_CONG bins × hour),
  and add week-of-year load factors.
- Weighted blending of bag predictions by their internal holdout AUC, and a second-stage logistic
  stacker with bag-embedding features rather than raw predictions.
- Calibrate decision to the hidden set's class balance (eval and train are both exactly 0.5 positive,
  which is almost certainly an artifact of the 100k sampling, not true of the 1M holdout).
