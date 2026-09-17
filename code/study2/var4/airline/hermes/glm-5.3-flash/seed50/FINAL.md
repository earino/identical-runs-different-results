# FINAL — autoresearch XGBoost (airline, scenario 2)

Best Eval AUC: **0.7433** (baseline 0.7141, +0.0292). Best commit: HEAD (`v40`).
All 40 experiments used; ~9,250 of 18,000 CPU-seconds; ~62 of 230 minutes.

## Final model

11-member XGBoost ensemble over interaction categoricals (all levels fitted on train.csv only;
unseen levels → NaN — the eval year contains carriers/airports absent from train, so this matters):

- Base features: DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest (categorical),
  DepTime, Distance (numeric). **Month is deliberately dropped** (2005 seasonality anti-transfers).
- Interaction categoricals: Origin×(20-min dep-time block), Carrier×(30-min block),
  Carrier/Dest/DOW×hour, Origin/Carrier/Dest/DOW×Distance-block, Distance-block×hour,
  Distance-block×(20-min block).
- Members: two full-set members (d5-n300-λ800 and d8-n200-λ120, lr 0.1); the six
  subset roles {dist, hour, base, carrier, origin} at d5-n600-λ200 / d8-n350-λ30;
  three slow-lr subset members (d5-lr0.05-n300-λ400 on dist/carrier/hour subsets).
  Predictions averaged.

## Changes that mattered most

1. **Strong regularization** (depth 5, mcw 20, λ200+): +0.003 solo. Train is 2005, eval is 2006 —
   more trees/depth overfit the training year (30 trees beat 400 on the baseline features).
2. **Interaction categoricals with train-fitted levels** (Origin×hour first, then Carrier/Dest/DOW×hour,
   ×Distance-block, ×finer time blocks): +0.010 cumulatively. The signal is in conditional operational
   patterns (which airport at which time, which carrier on which haul), not in global aggregates.
3. **Feature-subset ensemble** (dist/hour/base/carrier/origin members alongside full-set members):
   +0.008 — the biggest single late gain. Diversity of *inputs* beat diversity of seeds (seed
   averaging gave exactly nothing; ±0.0002 noise).
4. **Dropping the Month categorical** (2005 seasonality anti-transfers to 2006): +0.0016 solo.
5. **Trees, trees, trees for subset members**: n400→n600→n800 for the d5 subset roles kept paying
   (+0.0010, then +0.0009, then +0.0005); slow-lr (0.05) full members + heavy λ (800/120): +0.0010
   combined. Under regularization, the year-shift penalty fades with more boosting steps.

## What did not help

- Target encoding (OOF or smoothed), frequency features, route categoricals (Origin×Dest etc.),
  Month×DayOfMonth: all hurt — 2005 delay rates don't transfer to 2006.
- Deeper trees without λ, lossguide, max_bin changes, gamma, subsample, colsample_bytree,
  recency sample weights, row bagging, seed bagging: all neutral or worse.
- OOF stacking (meta-XGBoost on out-of-fold member predictions): −0.001 vs the simple mean.
- More full-set members (d6/d4/d7 configs), reweighting the mean, colsample diversity,
  adding members beyond ~11: all within ±0.0003 noise.

## With more budget

I would (1) continue the subset-member n-scaling (n800 was still improving), (2) map per-feature
time-resolution systematically at *ensemble* level (solo probes mislead — composition effects are
real), (3) run a randomized search over (config × feature-subset) pairs with greedy pruning on
multi-seed-averaged AUC, and (4) test a second-level mean of independently seeded full ensembles.
