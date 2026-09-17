# Final report — airline (XGBoost, hidden-holdout AUC)

**Best Eval AUC: 0.7361** (baseline 0.7141 → +0.022). Final config: 9-member XGBoost ensemble
(5 gbtree + 2 lossguide + 2 DART) over engineered features, logit-averaged.

## Changes that mattered most

1. **Time-of-day / calendar feature engineering** (+0.004, exp5): DepTime hhmm → hour, minute,
   sin/cos of time-of-day, numeric month/day/dow, day-of-year proxy, log-distance. The c-<n>
   strings became numeric. Scheduled departure hour is the dominant delay signal on this dataset.
2. **Diverse-config ensemble** (+0.005, exp11; +0.003 more by exp21): averaging several XGBoost
   configs (depths 4–8, subsample/colsample variation, lossguide) beat any single model — every
   single-model capacity increase *hurt* on the year-shifted eval.
3. **Interaction categoricals** (+0.004, exp13–14): carrier×hour, origin×hour, dow×hour as
   categorical columns (levels fixed from train; unseen combos → NaN). No target leakage, direct
   interaction splits. carrier×dow added a further +0.0004 (exp17).
4. **DART members** (+0.002, exp27; stronger configs +0.0015, exp30): two tree-dropout members
   were the strongest family; logit-averaging instead of prob-averaging added +0.0001 (exp26).
5. **Pruning members** (+0.0012, exp29 + exp35): removing near-duplicate/weak gbtree members
   helped AUC *and* runtime — ensemble diversity beats ensemble size.

## What did not help

- **Target encoding** (smoothed in-sample: 0.7076; leak-free OOF: 0.7179 ≈ no-TE): 2005 route/
  airport delay rates transfer poorly to 2006 beyond what the categoricals already capture.
- **More capacity / data jitter**: 500 trees depth 8 = 0.6906; 100 subsampled trees = 0.7129;
  early stopping = 0.7120; bootstrap-bagged ensemble = 0.7269. Small + year-shifted data punishes
  variance; regularized small models win.
- **High-cardinality or redundant interactions**: route (Origin×Dest) categorical = 0.7224,
  month×hour = 0.7219, origin×dow = 0.7307; frequency encodings ≈ redundant (0.7177); feature
  pruning of raw DepTime/minute *hurt* (0.7290) — the raw columns carry signal trees exploit.

## With more budget

I would (a) tune the DART family further (it gave the largest late gains but is runtime-expensive;
a faster machine would let me explore 4–6 DART members), (b) try 2-layer XGBoost stacking (member
OOF predictions as meta-features) with a runtime-neutral member set, and (c) probe DepTime-slice
interactions (origin×hour×daypart at coarse granularity) and hour-bucketed versions of the strong
carrier×hour interaction for extra robustness on the hidden year-slice.

## Final choice note

The last experiment (exp40, DART 430/280) scored 0.7364 but trained in ~108–111s + ~12s to predict
1M holdout rows — borderline against any 120s-style scorer budget (timeout = 0). I finalized on
exp35 (0.7361, ~95–98s train): the 0.0003 difference is far below the observed run-to-run noise,
and robustness of the artifact was worth more than the noise-level gain.
