# Final Report — airline delay (XGBoost, harness benchmark edition)

**Best Eval AUC: 0.7497** (experiments #22/#23; HEAD = `269c12e`, validated `CONTRACT OK`).
Baseline (experiment #1): 0.7141 → **+0.0356 AUC** over baseline.

Final model: 7-member XGBoost ensemble (seed bagging), each member `max_depth=7, lr=0.05,
colsample_bytree=0.7, reg_alpha=2.0, tree_method=hist, enable_categorical=True`, early-stopped
on eval.csv (AUC, patience 100), predictions averaged (arithmetic; logit-averaging scored identical).

## Changes that mattered most

1. **Early stopping on eval.csv as the validation set** (2006 slice1 as proxy for the hidden 2006
   holdout). This exposed the core difficulty: train is 2005, eval/holdout are 2006, so large models
   overfit 2005-specific patterns. ES consistently picked tiny capacities (~30 trees at lr=0.1) and
   later, with regularization, moderate ones (~700 trees at lr=0.05).
2. **Regularization shift toward robustness**: `max_depth 6→4→7` with `reg_alpha 8→2` and
   `colsample_bytree=0.7` (L1 sparsity + column subsampling = +0.006 over the ES baseline). Shallow-ish
   trees with strong L1 transferred best across the year shift; alpha>8 hurt once features improved.
3. **Interaction features as raw categoricals (biggest FE win, +0.017)**: `Origin×DepHour` (airport
   congestion by time of day) and `Carrier×DepHour` (carrier time-of-day banks), then
   `DistBin(log Distance)×Carrier` and `DepHour×DistBin` (haul-type effects). With
   `enable_categorical`, XGBoost partitions these high-cardinality columns natively.
4. **Pruning redundant features** (+0.002): dropping `Route` (Origin_Dest pair; covered by
   Origin×Hour/Dest) and `DayofMonth` (noise) while keeping `DepMinute` (slot effects, −0.008 if dropped).
5. **Seed-bagging ensemble** (5→6→7 members, +0.002 total): singles ~0.7475, 7-seed average 0.7493–0.7497;
   the seed-averaging curve flattens at 5–6 members (8 members would also breach the 120 s timeout).

## What did not help

1. **Target encoding** (OOF, smoothed k=20/100/300 for Origin/Dest/Route/Carrier/hour and pairs):
   hurt consistently (−0.01 at low capacity; still neutral-to-negative under strong L1) — noisy 2005
   group means do not transfer to 2006 better than native categorical partitions.
2. **Row subsampling** (`subsample=0.7–0.9`): always hurt (~−0.002); column subsampling helped instead.
3. **Extra numeric encodings**: `SlotDev` (distance to :00/:30 slot), `MinBin×Hour`, `Month×Hour`,
   `Origin×DoW`, `Origin×Month`, freq/count encodings, arrival-hour features (`Dest×ArrHour`),
   hour-wrapping of DepTime>2359 — all neutral or negative.

## With more budget

I would (a) enlarge the ensemble with cheap low-depth members and tune per-member learning rates for
diversity, (b) run a broader Bayesian-style sweep over the interaction set (e.g., Dest×Hour, finer
distance bins, airport-size classes) under the final regularized regime, and (c) investigate the
2005→2006 shift directly — e.g., importance-weighted validation or training-time reweighting of
patterns that persisted from 2005 slice to 2006 slice — since capacity, not features, is now the
binding constraint (every added feature past the interaction set was neutral).
