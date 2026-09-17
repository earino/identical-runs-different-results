# autoresearch XGBoost — final report (airline, 2005 → 2006 temporal split)

**Best Eval AUC: 0.7225** (baseline 0.7141, +0.0084). 40/40 experiments used, 2,620 of 18,000 CPU-seconds,
~1 h 20 m of the 230-minute wall clock. `HEAD` = `c4d8df8` ("reg_lambda 10.0"), which is the best kept commit.
`./validate.sh` → **CONTRACT OK** (train.py runs in 50.7 s, and `predict_proba` on `data/eval.csv` with the
target column removed reproduces 0.7225).

## Final model

- **Features** (all engineered inside `prepare()`, so `predict_proba` reproduces them on new rows):
  - categoricals fit on train only (unseen levels → NaN): `Month`, `DayOfWeek`, `UniqueCarrier`, `Origin`, `Dest`;
  - coarse time: `Hour`, `TOD = hour*60+minute`, `IsWeekend` (raw hhmm `DepTime` removed), plus sin/cos of hour and month;
  - `Distance`;
  - traffic-load family built from the training-year flight network only: volumes (`OriginVol`, `DestVol`, `RouteVol`,
    `OriginHourVol`, `DestHourVol`, `CarrierVol`, `HourVol`, `CarrierRouteVol`, `OriginCarrierVol`) and scale-free
    shares/ratios derived from them (`OriginHourShare`, `RouteShare`, `CarrierRouteShare`, `DestHourShare`,
    `OriginCarrierShare`, `OriginHourPeak`, `HourVolShare`).
- **Model**: 60-member XGBoost ensemble, `hist`, `enable_categorical`, 150 trees, `lr=0.1`, `reg_lambda=10`,
  `max_depth` ∈ {3,4,5,6} × 15 seeds, `subsample` ∈ {0.65…0.80}, `colsample_bytree` ∈ {0.50…0.85}; predictions
  are the mean of member probabilities. Train time ≈ 50 s, well inside the 120 s cap.

## The 5 changes that mattered most

1. **Capacity control — the single biggest lever.** The baseline regime (30 trees, depth 6) beat every bigger fit:
   300 trees depth 7 → 0.7045, 300 trees depth 6 → 0.7083, 500 trees depth 4 → 0.7105, depth 3 → 0.7119.
   A depth-4 / 150-tree point (0.7150 → 0.7156) is where eval AUC peaks; an early-stopping probe on a held-out
   15% of the training year independently picked **154 rounds**, confirming that budget. The 2005→2006 shift means
   extra capacity memorises the earlier year and transfers worse.
2. **Averaging diverse members.** 5 → 8 (depth-diverse) → 20 → 40 → 60 members moved 0.7167 → 0.7180 → 0.7184 →
   0.7215 → 0.7216. Rising ensemble size with varied depth/subsample/colsample was the most reliable gain.
3. **Traffic-load features** (volumes then shares/peak ratios): 0.7180 → 0.7175 (volumes) and later
   0.7190 → 0.7199 → 0.7210 → 0.7219 (shares, hour/dest-hour load, peak-shape ratios). These describe *who flies
   where and when* — network structure that is roughly stable across the year boundary, unlike anything derived
   from the target.
4. **Coarse time features instead of raw hhmm `DepTime`** (`Hour`, `TOD`, `IsWeekend`): cleaner, year-independent
   handles, and they free the shallow trees from fitting minute-level pockets.
5. **Dropping `DayofMonth`** (31 levels of pure noise): 0.7184 → 0.7190, an improvement *and* a simplification.

## Three things that did not help

1. **Smoothed delay-rate / target encodings** of Origin/Dest/Route/Carrier/Origin-hour (prior-corrected means,
   α=30): 0.7175 → **0.7057**. The worst single result. Delay *rates* of 2005 simply do not carry to 2006, and the
   model over-trusts them. Volume counts of the same keys help; rates of the same keys hurt.
2. **High-cardinality categoricals and interactions**: `Route` (4198 levels) as a direct categorical → 0.7021;
   `Month×Hour` (288 levels) → 0.7166. Both overfit the training year.
3. **Extra regularisation knobs / aggregation variants with no mechanism**: `max_bin=64` → 0.7179,
   `min_child_weight=20` → 0.7158, per-depth tree budgets → 0.7211, row bagging (80% per member) → 0.7223,
   rank-average instead of probability-average aggregation → 0.7215, and `reg_lambda=20` → 0.7225 (a tie with
   `reg_lambda=10`, so the simpler value was kept). Each stayed within the ~0.0016 single-run noise band of a
   100k-row AUC, and none of them earned their extra complexity.

## What I would try with more budget

The decisive fact about this task is that eval is a *different year*, so every gain came from making the model
less able to memorise 2005 — shallower trees, heavy averaging, and features that describe network/time structure
rather than outcomes. With more experiments I would (a) push the load/share family further, since it was the only
feature family still producing gains at the end — e.g. route gravity residuals `RouteVol / (OriginVol·DestVol)`,
carrier-at-origin-hour load, and consecutive-hour load deltas that proxy delay accumulation through the day;
(b) replace the single-point capacity choice with per-member early stopping on an internal 2006-shaped validation
slice, so each depth finds its own round count instead of my hand-picked 150; (c) explore gradient-boosted
stumps (`max_depth=2/3`, many rounds, stronger `reg_lambda`) as an even lower-variance member type, and grow the
ensemble into the hundreds while measuring whether the plateau at 40→60 members is truly flat; (d) validate the
final choice on a proper year-holdout so keep/discard decisions stop being made on 100k rows whose AUC has an SE
of ≈0.0016 — many decisions in this run were inside that band. One honest caveat: the four sin/cos cyclic features
were introduced in experiment 22 (0.7178, no better at the time) and, because a revert was skipped, were carried
through the remaining runs; the validated 0.7225 includes them, and I did not have an experiment left to ablate
them away, so they may be neutral baggage rather than a contributor.
