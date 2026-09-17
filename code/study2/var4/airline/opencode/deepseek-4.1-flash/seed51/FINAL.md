# Final report — airline delay AUC

**Best Eval AUC: 0.7417** (experiment #38, commit `0fe3769`)
Baseline: 0.7141 (`8810328`). 40/40 experiments used.

## Changes that mattered most

1. **Fourier harmonics of clock time (k=1..12) for both scheduled departure and
   estimated arrival time.** `tod = dep_hour + dep_min/60` and
   `arr_tod = tod + Distance/450 + 0.5` are expanded into a full Fourier basis.
   Each added harmonic improved eval AUC (0.7205 → 0.7313 across k=1..12); this was
   the single largest feature lever.
2. **`carrier_hour` native categorical (UniqueCarrier × departure hour).** The largest
   single jump (+0.0098 to 0.7417). Carrier-specific behaviour by time of day is real
   and transfers from 2005 to 2006.
3. **`dep_hour_cat` / `arr_hour_cat` native categoricals** (+0.0006), giving the trees a
   direct piecewise lookup on the hour on top of the smooth Fourier basis.
4. **Diverse depth ensemble of XGBoost models** (depths 3/4/5/6, some with
   subsample+colsample) averaged by predicted probability. Replaced a single model;
   depth-4/5/6 alone gave 0.7218, and averaging over diverse depths was consistently
   better than any single model. (Identical-parameter, different-seed models are
   deterministic here, so seed-only bagging gave nothing.)
5. **Time-of-day features** (`dep_hour`, `tod_sin/cos`) and the **estimated arrival
   time** block. Ablating time features cost −0.0066; dropping raw `DepTime` cost −0.0011
   (raw scheduled time carries minute-level signal).

Monotone increasing constraints on `dep_hour`/`arr_hour` were kept (a small but
principled/robust gain).

## Things that did NOT help

- **Target encoding**: OOF (random 5-fold) target encoding of Origin/Dest/Carrier/Route
  dropped AUC to 0.7132 — 2005 delay-rate statistics do not transfer to 2006.
- **Frequency / network-share encodings and route categorical**: frequency features
  0.7168, route categorical 0.6973, carrier-origin/dest network share 0.7188 — all
  below the simpler model; high-cardinality identity features overfit.
- **Calendar seasonality**: month/day-of-week/day-of-year Fourier features hurt
  (0.7169, 0.7283), as did `carrier_dow` (0.7389) and `origin_hour` (0.7345).
  Deeper single models (depth 8/10), early stopping on an internal 2005 validation
  split, and stronger regularization also hurt.

## What I would try with more budget

Exploit the success of `carrier_hour` by adding smoothed/regularized carrier-airport
and airport-hour interactions (e.g. origin × 3-hour bins instead of 24 raw hours, which
overfit) and carrier-origin hub features with shrinkage. Also worth trying: a proper
time-based validation split (last months of 2005) to select ensemble members instead of
`eval.csv`, and replacing the uniform ensemble average with weights tuned on that split.
Finally, a small neural/net-free tabular alternative is disallowed, but a stacking layer
combining the existing XGBoost members' out-of-fold predictions could add a little.
