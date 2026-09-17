# Final report — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7466** (baseline: 0.7141, +0.0325). Final `train.py` = 10-model XGBoost ensemble
(max_depth 3–7, 2 seeds each, equal-weight probability average).

## What mattered most

1. **Carrier × 30-min-slot interaction** (`carrier_hm30`, 480 levels): the single biggest feature gain
   (~+0.02 together with time features). Flight-delay risk is driven by carrier-specific schedule
   patterns, which are stable across the 2005→2006 year boundary.
2. **Aggressive regularization against the time shift**: shallow trees (depth 3–5 individually best),
   lr 0.03 × 2000 trees, reg_alpha 1, subsample 0.9. Deeper/faster single models overfit 2005-specific
   noise and lost up to 0.01 AUC on 2006.
3. **Time-slot categoricals**: `hm15cat` (96-level quarter-hour slot) +0.0016; hour/minute numerics
   +0.002. DepTime mod 2400 handles the wrap at midnight (values up to 2620 exist).
4. **Depth-diverse, feature-subset-diverse ensemble** (colsample_bynode 0.5): 6-model d3/d4/d5 mix
   +0.0017 over the best single model; adding d6×2 and d7×2 members pushed to 0.7466 (+0.0010 more).
5. **colsample_bynode 0.5** (feature-subset diversity per split): +0.001 in the ensemble; much better
   diversity knob than row subsampling or seed count alone.

## What did not help (all tested, all reverted or skipped)

- Target encoding of carrier/origin/dest/route (smoothed, train-fitted): −0.005 to −0.03 — redundant
  with categorical splits and hurt by the year shift.
- High-cardinality categorical interactions: route (Origin_Dest, 4.2k levels), origin_hour, dest_hour,
  carrier_hour_dow, dow_hm15, month×anything — all substantially negative (unseen-level fragility).
- Bigger single models (depth 6–8, 500–3000 trees, lr 0.06–0.1), early stopping on a random split
  (stopped at ~123 trees, underfit), recency sample-weighting, row-bagging, rank averaging,
  frequency/count features, extra time-bin categoricals (hm5/hm10/hm30cat), carrier at minute
  granularity, per-depth lr/n tuning, ensemble weight tuning (best grid point = +0.0001 = noise).

## With more budget

I would (a) test gradient-boosted members trained on bootstrap-resampled 2005 data weighted toward
late-2005 months for a gentler distribution shift, (b) grow the ensemble along the depth axis with
early-stopped per-member tree counts chosen on an internal 85/15 split, and (c) revisit out-of-fold
target encoding only for the route pair, computed strictly within 2005 and smoothed hard — the one
feature family that failed here but works on static splits. Everything else was saturated: the last
six changes moved eval AUC by less than the ±0.0002 seed noise.
