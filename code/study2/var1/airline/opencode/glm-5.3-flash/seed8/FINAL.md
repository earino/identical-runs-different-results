# Final report — airline delay AUC

**Best Eval AUC: 0.7480** (baseline 0.7141, +0.0339)

Final model: bagged ensemble of 6 XGBoost models (seeds 42–47), each `max_depth=18, n_estimators=100,
learning_rate=0.1, subsample=1.0, colsample_bytree=0.8, max_bin=512, hist`, on 6 raw features
(DayOfWeek, DepTime, UniqueCarrier, Origin, Dest, Distance as categoricals/numeric) plus sin/cos of
time-of-day. Month and DayofMonth are deliberately dropped.

## Changes that mattered most

1. **Dropping Month and DayofMonth** (+0.0033 at d4): train is 2005, eval 2006; these columns carried
   mostly year-specific noise that trees memorized. Diagnosed via 2005→2006 rate correlations
   (DayOfWeek 0.96, carrier 0.85, but DayofMonth 0.33) and confirmed by ablation.
2. **Deep trees on stable features** (d6→d18: 0.7237→0.7449): once noise columns were removed, AUC rose
   monotonically with depth — stable features support deep interactions that transfer across years.
   Early stopping on a 2005 validation split was actively harmful (it always picked far too many trees);
   fixed small ensembles beat probe-based selection.
3. **Sin/cos encoding of DepTime** (+0.0014): cyclical time-of-day captures the midnight wraparound
   that raw hhmm cannot; helped only after Month/DayofMonth were removed.
4. **Seed bagging** (K=8, +0.0011 at d4n80): variance reduction without extra effective capacity.
5. **Subsample 1.0 + colsample 0.8 + max_bin 512 at d18** (+0.0031 total): row-subsampling hurt deep
   trees (0.8→1.0 monotone gain), column subsampling was essential (1.0 collapsed to 0.7332), and finer
   histogram bins gave a last +0.0007.

## What did not help

- **Target encodings** (carrier, route, origin/dest, carrier×hour, dow×hour; tried 4× with different
  smoothing and leakage controls): always hurt, e.g. 0.7021/0.6957/0.7053/0.7106 vs ~0.714-0.722
  baselines. Per-group 2005 rates are too noisy/shifted for 2006 (route-rate correlation 0.28).
- **Time-of-day as hour categorical / late-night flag** at shallow depth: duplicated raw DepTime and
  consumed capacity (0.7151 vs 0.7157).
- **Structural extras**: traffic counts + per-carrier distance stats (0.7169, equal), route categorical
  (0.7083, catastrophic), mixed-depth/param ensemble diversity (0.7217), max_bin 64 (0.7189),
  gamma/min_child_weight regularization at depth (worse), dropping Distance/Dest (worse at depth).

## With more budget

- Finer joint sweep around the final config (depth 16–20 × n 80–120 × max_bin 256/512/1024 ×
  colsample 0.7–0.9) with more seeds per point to beat the ±0.001 selection noise.
- A second feature axis: flight-level aggregates that are year-stable by construction (e.g. per-airport
  hourly traffic pressure) rather than target-derived ones, which repeatedly failed.
- Larger bag (K≈24) of the d18 config if runtime allowed, and rank-averaging instead of mean-averaging.
