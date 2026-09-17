# Final report — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7332** (baseline 0.7141, +0.0191) — `train.py` at HEAD (`c073d0d`), validated `CONTRACT OK`.

## Final model

Ensemble of 4 XGBoost models (average of predicted probabilities), each trained on 100% of
`data/train.csv` with 1413 trees (lr 0.05, hist, native categoricals):

| member | depth | subsample | colsample_bytree | gamma |
|--------|-------|-----------|------------------|-------|
| 1 | 8 | 0.8 | 0.8 | — |
| 2 | 12 | 0.7 | 0.9 | 1.0 |
| 3 | 7 | 0.9 | 0.7 | — |
| 4 | 17 | 0.8 | 0.8 | 3.0 |

Features (all engineering inside `prepare()`, encoders fit on train only): parsed Month/DayOfMonth/DayOfWeek
ints, DepTime → hour/minute/sin/cos + raw, Distance + log, native categoricals for carrier/origin/dest,
frequency encodings, route frequency, hub-ness (distinct destinations per origin / origins per destination /
origins per carrier), route distance deviation.

## Changes that mattered most

1. **Time-of-day feature engineering** (exp 3, 0.7141 → 0.7232): parsing `c-N` strings to ints, DepTime →
   hour/minute + cyclic encoding. The single biggest jump; raw `hhmm` integer and string months waste signal.
2. **Full-data-refit ensemble** (exp 10–11, → 0.7255): members early-stopped on a shared 80/20 split, then
   refit on 100% of train at that iteration count; bagging on 80% subsets was strictly worse (exp 9).
3. **Structural features: hub-ness + route stats** (exp 12, → 0.7280): distinct-destination counts and
   route-frequency / route-distance-deviation. Target-free, so robust to the 2005→2006 drift.
4. **A deep, gamma-regularized member, then two** (exp 31–40, 0.7280 → 0.7332): one member pushed to
   max_depth 17 with gamma 3.0 (ratcheting depth with pruning was monotone-positive for 6 straight steps),
   and replacing a redundant second depth-8 member with a second deep member (d12, gamma 1.0) gave the
   final +0.0015.
5. **Hardcoded per-config tree counts** (exp 27): early stopping is deterministic for this data/feature set,
   so hardcoding the ES-selected count (1413) removed a ~20–30s ES sweep per run and enabled the ensemble
   scaling experiments within the 120s cap.

## Things that did not help

1. **Target encoding in any form** (exp 4, 5, 15/16): airport/carrier/route TE and even coarse month/dow TE
   with OOF overlays — every variant inflated 2005 val AUC but hurt 2006 eval; the year gap drifts group
   delay rates too much.
2. **Capacity without regularization** (exp 2, 6, 7): more trees past ~1400–1800, depth 6 + min_child_weight,
   max_bin 512 — all flat or worse; the eval set punishes plain extra capacity.
3. **Small structural extras**: route frequency + DepTime anomaly flags (exp 8), congestion counts
   (exp 17–19, 30), colsample_bynode diversity (exp 29), a 5th homogeneous member (exp 28) — all exactly
   neutral (0.7280) while adding code, so reverted per the simplicity criterion.

## With more budget

I would push the deep-member frontier harder: a 3rd deep member (d12–d15, varied gamma/colsample), per-member
gamma/depth ratchets, and per-member tree counts re-derived by ES on full data (per-member 80%-ES counts
underestimated and hurt — exp 35). I would also revisit interaction features that survived the drift filter
(e.g., carrier × hour-bank structure) and probe larger subsample diversity (row-bagged members on fixed 90%
row subsets). Given the 120s-per-run cap, the ES sweep would stay hardcoded; the CPU budget (~13.8k of 18k
used) was never binding, but wall-clock per run was the real constraint.
