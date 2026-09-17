# FINAL — airline delay XGBoost (szilard scenario 2)

**Best Eval AUC: 0.7443** (commit `590d9ff`, experiment #39; baseline was 0.7141).

## What mattered most

1. **Dense interaction categoricals** — the single biggest lever (+0.022 combined):
   - `Carrier×DepHour` (~450 levels): exp #28, 0.7221 → **0.7357**
   - `Carrier×DistBin` (~140 levels): exp #33, 0.7357 → **0.7377**
   - `DepHour×DistBin` (~170 levels): exp #35, 0.7377 → **0.7427**
   Only *dense* interactions (≥ ~100 rows/level) helped; sparse ones (route, origin×hour,
   carrier×origin, carrier×month) actively hurt by -0.005 to -0.014.
2. **Diverse 11-member XGBoost ensemble averaged** (exp #12–#25, 0.7166 → 0.7221):
   depths 6–9 hist + 3 lossguide (leaves 32/64/128) + 2 dart members; one shared early-stopping
   round count (stratified 80/20 split, es=30, then refit all members on full train at best_n).
   Boosting-algorithm diversity (dart/lossguide) gave real gains; row-subsample bagging did not.
3. **DepTime parsing** (exp #4): hour-of-day category + sin/cos of minute-of-day. Removing the
   sin/cos dropped AUC by 0.027 (exp #31) — continuous time signal is essential.
4. **Early stopping + refit** (exp #5): 30-tree baseline → ES-selected ~575 rounds, lr 0.05,
   depth 8: 0.7147 → 0.7166.
5. **Coarse distance bins** (exp #39): 10 → 7 bins gave 0.7427 → 0.7443 (denser interaction
   levels); finer 15 bins hurt.

## What did not help

- **Target encodings** (in-sample smoothed, exp #7: -0.0004; OOF 5-fold, exp #15: -0.0026).
- **High-cardinality raw categoricals**: DepTime as cat (0.7027), route/carrier-origin/carrier-dest
  (exp #6: 0.7050), origin×hour (timeout), carrier×month (exp #34: 0.7236).
- **Bagging** 80% row subsample per member (exp #17: 0.7185 vs 0.7193) and val-AUC-weighted blending
  with in-sample scores (exp #20: equal).

## With more budget

I would (a) build a small grid of interaction families (carrier×quarter-hour, origin×dow on hub
airports only) selected by a proper time-aware CV, (b) tune per-archetype round counts (dart and
lossguide currently share the hist ES round count), and (c) grow the ensemble to 15+ members after
cutting per-member cost (smaller finder, n_jobs accounting), since every added *diverse* member
helped while same-archetype seeds plateaued. True out-of-fold stacking with an XGBoost meta-learner
is the other untried direction.

## Reproduce

`./run_experiment.sh` on commit `590d9ff`; validation: `CONTRACT OK`, eval AUC via `predict_proba`
= 0.7443. Final model trains in ~68 s on 4 threads, well within the 120 s / 6 GB limits.
