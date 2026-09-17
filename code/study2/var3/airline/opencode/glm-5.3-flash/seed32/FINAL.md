# FINAL — airline delay XGBoost

**Best Eval AUC: 0.7366** (baseline 0.7141, +0.0225). Final config: 12-member XGBoost ensemble
(depths 2–7 + lossguide, lr 0.03–0.1) over engineered features, all training on the full 100k rows.

## Changes that mattered most

1. **Time-of-day × flight-distance interaction categorical** (half-hour blocks 0–47 × 10 distance
   quantile bins, and hour 0–23 × same): the single biggest win (+~0.010 AUC alone). The hhmm integer
   DepTime and raw Distance could not express "short-hop red-eye vs transcon evening" delay profiles.
2. **More low-cardinality interaction categoricals**: hour×carrier (24×20 — ablating it cost −0.0095),
   month×hour, dist_bin×carrier. Stable across the 2005→2006 year gap, unlike high-cardinality ones.
3. **Diverse shallow ensemble** (12 members: d2–d7, lossguide max_leaves 24/64, lr 0.03–0.1): +~0.002
   over the best single model (d4/800/lr0.05 ≈ 0.7210 solo vs 0.7366 ensemble).
4. **Cyclical time features + hour categorical + distance quantile bins** (sin/cos of hour, month,
   day-of-month, day-of-week; log1p distance): +~0.005 over raw columns.
5. **Shallow-capacity regime**: depth 3–4 with lr 0.05 and 800–1200 trees beat deeper models at every
   step; max_bin=512 helped marginally.

## Things that did not help

1. **Target encoding** (K-fold smoothed, Origin/Dest/carrier/Month): 0.7194 vs 0.7210 — cross-year level
   statistics don't transfer; per-level memorization is penalized by the time split.
2. **Row/feature reduction of any kind**: bootstrap bagging (0.7203), subsample/colsample 0.8 (0.7125) —
   the model wants all 100k rows and all features; diversity must come from hyperparameters.
3. **High-cardinality raw combos**: route Origin_Dest categorical (0.6956), half-hour×carrier 960 levels
   (0.7288) — too sparse/unstable across years. Also failed: early stopping on a random 2005 split
   (val AUC 0.60 wildly underestimates 2006 eval AUC), rank-averaging, weighted ensembling, more dist bins.

## Theory of the data

Only ~8 raw columns, so nearly all signal is in stable aggregates: time-of-day (dominant), season,
carrier, distance band, and especially their intersections. The 2005→2006 shift punishes any feature
that memorizes per-level rates; it rewards coarse, smooth, interaction-based structure. Ensemble
diversity via depth/growth-policy works; diversity via data perturbation does not.

## With more budget

- Grid over the hour×dist interaction granularity (e.g., 20-minute blocks, more distance bins per-cell
  with count-based pruning), and a 2-way vs 3-way interaction search (hour×dist×carrier with pruning).
- Origin-level aggregates that survive the year gap (e.g., origin size/hub-ness bins rather than raw
  origin interactions).
- Per-member feature subsets chosen by stability across a 2005-holdout split, and a larger ensemble
  with per-member time budgets (current run: ~99s of the 120s limit).
