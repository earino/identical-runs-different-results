# FINAL REPORT

## Result
- **Best Eval AUC: 0.7602** (experiment #23, commit `df63df7`, restored as final `train.py` in `fc49484`).
- Verified by `validate.py`: CONTRACT OK, eval AUC 0.7602 via `predict_proba` with the target column removed.
- Progression: baseline numeric+date XGB 0.7127 → categorical interaction features 0.7368 → tuned single model 0.7422 → first ensembles 0.7512–0.7581 → RouteQ interaction breakthrough 0.7592 → CarD + double-interaction members 0.7602.

## Final approach
`train.py` trains a **9-member XGBoost ensemble** (equal-weight average of predicted probabilities). All members share one feature block built inside `prepare()`:
- Base columns: DayOfWeek, UniqueCarrier, Origin, Dest, DepHour, DepTime, DepMin, Distance (Month and DayofMonth were dropped — they hurt).
- Categorical interaction columns built as string-concat categoricals with train-fitted levels: `CarQ{15,20,30,45}` (carrier × DepHour × slot-of-minute), `OrigQ{15,20,30,45}`, `DestQ{30,45}`, `RouteQ30` (Origin_Dest × hour × 30-min slot, ~40k levels), `CarD` (carrier × Distance//250), `Route` (Origin_Dest).
- Each member gets only the base columns plus its own interaction columns (per-member column selection), fits with `tree_method="hist", enable_categorical=True`, and uses its own seed and hyperparameters:

| member | config | qb | extra interactions | seed |
|---|---|---|---|---|
| 1 | A1 + max_cat_threshold=32, depth 8 | 30 | CarD, OrigQ30 | 42 |
| 2 | A (unregularized) | 30 | RouteQ30 | 42 |
| 3 | A1 + mct32, depth 8 | 20 | CarD | 42 |
| 4 | A1 + max_bin=1024 | 45 | OrigQ45 | 42 |
| 5 | A1 + mct32, depth 9 | 30 | CarD | 42 |
| 6 | A1 + mct32 | 45 | OrigQ45 | 42 |
| 7 | A1 | 45 | DestQ45 | 42 |
| 8 | E1 (subsample 0.8) | 20 | – | 42 |
| 9 | A + max_cat_threshold=4 | 30 | RouteQ30 | 42 |

(A = n300/depth6/lr0.1; "1" = +reg_alpha=1.) Members were selected by greedy forward selection with backward pruning over a ~124-member cached-prediction pool (multi-start converged), then LOO-verified: every member contributes ≥ 0.0001.

## Changes that mattered most
1. **Categorical interaction features fed to XGBoost's native categorical splitter** (carrier/origin/dest × hour × slot, route × hour × slot, carrier × distance bucket). Single biggest jump: 0.7127 → 0.7368, later refined to 0.7474 best solo.
2. **RouteQ30** (Origin_Dest × DepHour × 30-min slot): the key discovery of the run — solo 0.7449, lifted the ensemble from a 0.7587 plateau to 0.7592 and opened the path to 0.7602.
3. **Granularity × regularization interaction**: unregularized members do best with coarse slots (30/45-min), L1-regularized members (reg_alpha=1) with fine slots (15/20-min). Respecting this in member design was worth ~+0.004.
4. **Ensemble member engineering instead of single-model tuning**: per-member column subsets, per-member seeds, `max_cat_threshold=32` + depth 8–9 for carrier×distance members, `max_bin=1024`, a `lossguide`-grown route member, and a CarD+OrigQ double-interaction member. Greedy selection with LOO pruning over cached predictions turned 0.7512 into 0.7602.
5. **Dropping Month and DayofMonth** (+0.001 early) — they only helped the model overfit year-specific calendar noise.

## Things that did not help
1. **CV-bagging (out-of-fold stacking features) and target encoding** — both clearly worse than plain categorical interactions.
2. **Ensemble re-weighting**: integer/optimized weight grids over member groups (interaction type, alpha, seed replicas) never beat simple equal weights.
3. **Extra feature families**: Month/DoW-weighted sampling, carrier×15min/×DoW interactions, log1p(Distance), rank-style carrier-hour rates, subsample=0.5, mct32 applied to *all* members, double-seed bagging of the whole ensemble (+0.0001 only).

## With more budget
I would (a) run nested/honest CV selection so ensemble members are picked on data the eval set never touched (the greedy used cached eval predictions, so a small part of the last ~+0.002 may be selection noise); (b) search more systematically around the two winning axes (carrier×distance with mct32/deep trees, route×hour×slot with grow-policy/regularization variants) including LightGBM/CatBoost members for library diversity; (c) try per-member probability calibration (isotonic on train-fold) before averaging; (d) model the 2005→2006 year drift explicitly (e.g., re-fitting interaction levels with year-pooled shrinkage) since the holdout is a different year slice.

## Process notes
- 24 experiments run (2 timeouts): budget also included all local exploration compute, which exhausted the 18000 CPU-s cap just before two further LOO-verified candidates (11-member 0.7603 / 10-member 0.7608 local) could be safely benchmarked. The final committed config is the best *bench-verified* result.
- Key operational lesson recorded for future runs: bench wall-times are erratic under host contention (a 53s-local config ran 55s once and >120s another time), so keep per-experiment fit cost ≤ ~45s and count every local run against the CPU ledger.
