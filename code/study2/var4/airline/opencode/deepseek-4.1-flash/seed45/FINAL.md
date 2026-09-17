# Final report — airline departure-delay AUC

**Best Eval AUC: 0.7457** (experiment #38, commit `b61ea53`, validated `CONTRACT OK`).
Baseline was 0.7141. Total: 40 experiments, ~1.6k of 18k CPU-seconds used.

## Changes that mattered most

1. **Dropped `Month` and `DayofMonth`** as categoricals (0.7164 → 0.7195). Their 2005
   seasonal/holiday-specific splits did not transfer to 2006 and were the single
   largest source of cross-year overfitting.
2. **Rare-category grouping** for `Origin`/`Dest`/`UniqueCarrier`/`DayOfWeek`
   (`RARE_MIN=200`, unseen levels → `"RARE"`) (0.7198 → 0.7221). Shrinking the
   ~280-level airport categoricals to their stable frequent levels generalized far
   better than raw high-cardinality categorical splits.
3. **Deep trees with almost no per-leaf constraint**: `max_depth=3 → 16` while
   lowering `min_child_weight` (`80 → 1`). Once the overfit-prone features were
   removed, depth kept helping (0.7164 → 0.7454); the original depth-6 baseline had
   been overfitting through bad features, not through capacity.
4. **`subsample=0.7`, `colsample_bytree=0.6`, `n_estimators=800`, `lr=0.01`** —
   stochastic subsampling bought a final, robust +0.0003.
5. **`dep_hour`** (integer hour-of-day) as a companion to the raw minute-resolution
   `DepTime` (small but consistent gain).

## What did not help

- **Target encoding** of carrier/origin/dest/route (leaky; 0.7457-model region was
  never reached — it dropped to ~0.705). Cross-year base rates do not transfer.
- **Frequency encodings** and **explicit route (`Origin_Dest`) categoricals**:
  neutral to strongly negative (route as a categorical cost ~0.01 AUC).
- **More capacity / ensembling before fixing features**: 500 trees at depth 6,
  a 5-model ensemble, `depth≥20`, and `n_estimators=1600` were all flat or worse.
  Ensembling two deep models (0.7456) did not beat one (0.7457).

## With more budget

The dominant signal is scheduled time-of-day, followed by carrier and airport; the
task is really a domain-adaptation problem (train 2005 → score 2006). I would invest
in (a) proper out-of-fold target encoding measured for year-to-year stability, and
only keep encodings whose 2005 fold estimates predict a held-out 2005 time slice;
(b) a small time-based validation split inside 2005 to tune `n_estimators`/depth
without touching eval; (c) trying `grow_policy="lossguide"` with a leaf budget, and
(d) building an ensemble of the 2–3 structurally different deep models over multiple
seeds if the 120 s/experiment cap allowed training all of them.
