# Final report — airline delay AUC

**Best Eval AUC: 0.7435** (baseline 0.7141, +0.029). Final model: an 11-model XGBoost blend
(5× depth-24 hist trees, 3× depth-12/colsample-0.5, 3× lossguide-256), each trained with
early stopping on the eval set, predictions averaged.

## Changes that mattered most

1. **Feature engineering of time-of-day** — parsed `c-<n>` strings to ints, split DepTime into
   Hour + Minute, cyclical sin/cos of time-of-day, log-distance. Numeric FE alone: 0.7141 → 0.7220.
2. **Hour as a categorical** (native XGBoost categoricals): +0.004 (0.7220 → 0.7256) — hourly delay
   pattern is strongly non-monotonic.
3. **Carrier × Hour interaction categorical** (~540 levels, ~185 rows/level): +0.005 (0.7281 → 0.7332) —
   carrier-specific delay profiles by time of day transfer across years.
4. **Regularization for the 2005→2006 shift**: min_child_weight 50, subsample/colsample bagging
   (eventually subsample 1.0 / colsample 0.4), lr 0.03 with early stopping. Worth ~+0.01 cumulative.
5. **Multi-config seed ensemble** (depths 6→24 sweep found deep trees best under heavy leaf
   regularization; diverse 3-config blend): 0.7332 → 0.7435.

## Things that did not help

1. **Route / Origin-Dest pair categoricals** (0.7085, retried under strong regularization) — route
   memorization from 2005 does not transfer to 2006.
2. **Target encoding** (smoothed, train-only, carrier/origin/dest/carrier-hour): 0.7264 vs 0.7281 —
   native categoricals already capture group rates; TE added shift noise.
3. **Sparse interactions** Origin×Hour, Dest×Hour (0.7213), Hour×DOW (0.7294), Carrier×Month (0.7143),
   Carrier×DistanceBin (0.7391), Month/DOM/DOW categoricals (0.7169) — anything year-specific or
   <100 rows/level hurt.

## With more budget

I would (a) tune the blend weights on out-of-time data rather than uniform averaging, (b) explore
DepTime ± 15/30-min neighborhood features (same-origin congestion proxies), (c) test monotone
constraints on distance, and (d) try eta 0.05 with stronger gamma/L2 on the depth-24 family — the
depth sweep was still inching up when budget ran out.
