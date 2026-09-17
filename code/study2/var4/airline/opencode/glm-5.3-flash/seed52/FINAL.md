# Final report

**Best Eval AUC: 0.7479** (commit a39bc89; baseline was 0.7141, +0.034). Contract validated (`CONTRACT OK`).

## Changes that mattered most

1. **Target encoding (smoothed, out-of-fold)** for high-cardinality keys — carrier/origin/dest/route plus
   interactions (carrier×hour, origin×hour, route×hour, dest×hour, carrier×day-of-week, 15-min departure
   slots, distance bins). This was the single biggest step (0.7141 → 0.7318 when combined with richer keys).
2. **Frequency (count) encoding** of the same categorical/interaction keys (normalized counts) — the largest
   single jump (0.7365 → 0.7426, then → 0.7463 with interaction keys). Popularity/traffic level of a route or
   airport transfers well across the 2005→2006 year shift.
3. **Recency sample weights** — linearly upweighting later months of 2005 (Jan=1 → Dec=6) because eval/holdout
   are from 2006: 0.7464 → 0.7479. Slope sweep (1/2/3/4/5/7/10) peaked at 5.
4. **3-model diverse XGBoost ensemble** (depth 6/7/8, colsample 0.9/0.8/0.6, 1200 trees, lr 0.03,
   subsample 0.8, reg_lambda 5): 0.7353 → 0.7365.
5. **TE smoothing m=10** (swept 5/10/20/50) with 5-fold shuffled OOF for training rows; full-train mapping
   applied to unseen data inside `prepare()`/`predict_proba()`.

## Things that did not help

- Month interaction TEs (origin/dest/carrier/route × month): 0.7433 — seasonality does not transfer well.
- TimeSeriesSplit (month-ordered) OOF for TE: 0.7424; 10-fold OOF: 0.7448 (5-fold shuffled is best).
- Extra TE keys (Slot15×DoW, CarrierDist, RouteDist, RouteCarrier, WeekOfYear), TE product features,
  route-typicality ratios (DistVsRoute/DistVsCarrier), min_child_weight=10, max_bin=512, a 4th/5th/6th
  ensemble member, and log-scaled counts — all equal or worse.

## With more budget

I would try: per-key adaptive TE smoothing (by cardinality), a larger/stacked ensemble with
out-of-fold meta-features, per-month dep-delay base rates as calendar features, quantile instead of fixed
distance bins, and a small random search over ensemble member configurations (depth/colsample/subsample/lr)
with rank-averaging instead of probability averaging.
