# Final report — airline delay (dep_delayed_15min)

**Best Eval AUC: 0.7365** (experiment #40, commit c0b4c9f). Budget exhausted: 40/40 experiments,
~23 min wall clock, ~3600/18000 Python CPU-seconds. `./validate.sh` prints **CONTRACT OK**.

## Changes that mattered most

1. **Out-of-fold target + frequency encoding of interaction keys** (biggest win, 0.7141 → 0.7240+).
   12 keys: carrier, origin, dest, route, carrier×route, origin×hour, carrier×hour, hour,
   origin×month, carrier×month, route×dow, dest×hour. Target encoding uses additive smoothing
   (k=5–30) and 5-fold out-of-fold values for the training rows; frequency counts capture traffic
   volume. This let the model use Origin/Dest without the severe overfitting of raw high-cardinality
   categoricals.
2. **Numeric time/seasonality features** (`dep_hour/min/tod` + sin/cos, `dayofyear`, `month`, `day`,
   `dow`, `is_weekend`). Adding the numeric seasonality block gave 0.7280 → 0.7297.
3. **`grow_policy="lossguide"` leaf-wise trees, `max_leaves=512`** (0.7299 → 0.7346). Leaf-wise growth
   was the single largest structural gain; gains kept coming as leaves rose 64 → 128 → 256 → 384 → 512.
4. **Bagged ensemble of 5 XGBoost seeds** averaged on predicted probability (≈ +0.001, and much more
   stable than any single seed).
5. **Regularized, slow-learning base learner**: 400 trees, lr 0.03, `colsample_bytree=0.5`,
   `min_child_weight=30`, `reg_lambda=5`, `reg_alpha=0.5`, `gamma=0.1`. Also ratio features
   (`route_share`, `origin/dest_carrier_share`) added 0.7365 on the last run.

## What did NOT help

- **More capacity on raw features**: depthwise 300–500 trees at depth 7 on the baseline features
  dropped AUC to ~0.703–0.714, and re-adding raw `Origin`/`Dest` categoricals cost ~0.003. The
  2005→2006 time shift punishes high-variance splits.
- **More interaction keys / encodings**: a second and third batch of TE keys (origin×carrier,
  route×month, origin×dow, hour×dow, month×dow, …) were neutral or negative; 10-fold OOF, raw
  `DepTime`, cyclic day-of-year/month, and altered smoothing were all within noise or worse.
- **Diverse-config and larger ensembles**: mixing depthwise + lossguide bags (0.7341), 10 identical
  seeds, and 10×5 diverse configs were all ≤ 5 identical lossguide bags.

## What I would try with more budget

Build an explicit **time-aware validation** (last slice of 2005, or 2006 slice1 used as a dev year)
to pick complexity instead of trusting the single eval slice, since eval AUC differences below
~0.002 are inside its noise. Then explore per-key smoothing tuned by that validation, target
encoding computed with a decay toward recent months, explicit year-over-year calibration of the
encoded rates, and richer spatial interactions (origin/dest station-level congestion, route-level
competition). A small stacking layer over the 5 bagged XGBoost models (still XGBoost-only) is the
most promising remaining structural idea.
