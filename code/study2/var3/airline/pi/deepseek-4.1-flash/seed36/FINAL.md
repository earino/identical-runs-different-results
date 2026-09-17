# Final report — airline delay AUC

**Best Eval AUC: 0.7603** (commit `0b1434e`, experiment #25). Baseline was 0.7141.
Final model validated with `./validate.sh` → `CONTRACT OK` (AUC 0.7603 via `predict_proba` on
`data/eval.csv` with the target column removed).

## Changes that mattered most

1. **Depth-diverse XGBoost ensemble (2 → 28), simple probability averaging.** This was by far the
   biggest win (0.7141 → ~0.7393, and it kept improving as the depth range grew to ~28: 0.756).
   Individually the deep trees overfit the 2005 training year badly, but averaging predictions over a
   wide spread of tree depths cancels a large amount of year-specific variance. The depth-2..5 members
   carry the stable signal; the very deep (20–28) members contribute diverse, high-variance error that
   averages out. Deep members use few trees (20–35) to stay inside the 120 s budget.
2. **Aggressive per-tree feature subsampling (`colsample_bytree=0.4`).** Lowering colsample from 1.0
   monotonically improved the ensemble (0.7462 → 0.7492 → 0.7528 → 0.7538 at 0.4; 0.25 was worse). The
   randomisation is the source of the useful diversity.
3. **Per-member random seeds (`random_state = SEED + 7*i`).** With `colsample_bytree<1`, the seed
   controls which features each tree sees; giving every depth member an independent seed was free and
   gained +0.0016 (0.7587 → 0.7603).
4. **Dropping `Month` and `DayofMonth`.** These encode 2005-specific seasonality that does not transfer
   across the time split; removing them gave +0.003 on a single model. `DayOfWeek` and `DepTime` stayed.
5. **Schedule-density features derived from `DepTime`:** hour/minute, cyclic sin/cos, and counts of
   `Origin×hour`, `Dest×hour`, `UniqueCarrier×hour`, `UniqueCarrier×Origin`, `UniqueCarrier×Dest`
   (frequency only, fitted on train). Together ~+0.006 in the ensemble.

## Things that did not help

* **More boosting at a fixed model shape** (30 → 300 → 500 trees, deeper single trees): consistently
  overfit 2005 and lowered eval AUC (0.7141 → 0.7070).
* **Target / mean-encoding of `Origin`, `Dest`, `UniqueCarrier`, `route`** (smoothed, train-only):
  large drop (0.7202 → 0.7042). The 2005 delay rates do not transfer to 2006.
* **High-cardinality categorical `route` (Origin_Dest), route-hour counts, origin×weekday counts,
  monotone constraints on `DepTime`:** all neutral-to-harmful (route categorical alone −0.013).
* **Row subsampling (`subsample=0.8`) and structural `lossguide` members:** no gain for the added
  complexity/time.
* Doubling the number of members at half trees was worse (deep members became too weak), and pushing
  the depth range to 32 exceeded the 120 s limit without improving AUC.

## What I would try with more budget

The dominant effect here is model diversity, not any single feature, so I would invest in richer and
more principled diversity. Concretely: (a) proper out-of-fold/target-encoded members blended into the
non-target ensemble, with encodings computed on rolling time windows rather than the whole training
year, which may recover carrier/airport signal without the year-shift failure seen here; (b) a stacked
meta-learner (small XGBoost) over the per-member out-of-fold predictions instead of a plain average,
which can learn to down-weight the deep, noisy members; (c) explicit multi-scale interaction features
(e.g. `DepTime` binned × hub indicators) to give the shallow, stable members more of the signal the
deep members currently supply only through variance; and (d) a wall-clock-aware schedule that spends
the full budget on as many independent, differently-seeded members as the 120 s cap allows, since
returns from adding depth diversity had not yet saturated at depth 28.
