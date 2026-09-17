# Autoresearch XGBoost — airline delay (final)

**Best Eval AUC: 0.7233** (experiment #40, commit `329002a`; contract validation via
`predict_proba` on target-stripped `eval.csv` also = 0.7233, `CONTRACT OK`).

Baseline was 0.7141 (30 trees, depth 6, lr 0.1). Final model is an 18-member XGBoost
ensemble over diverse (depth, colsample, subsample) settings, 900 trees at lr 0.02, with
strong regularization.

## Changes that mattered most
1. **Shallow, heavily regularized trees.** `max_depth=4` (vs 6/8), then `min_child_weight`
   10→80, `reg_lambda` 5→40, `reg_alpha` 1→2, `gamma=1`, `lr=0.02` with 900 trees. The
   2005→2006 time split punishes capacity; this ladder alone took 0.7141 → ~0.7219.
2. **Congestion *ratio* features (normalized frequency).** Counts of flights per
   `Origin`/`Dest` × departure-hour, and the fraction of that airport's/carrier's traffic at
   that hour. These robust, label-free activity features took 0.7183 → 0.7196 and beyond.
3. **Estimated arrival-time features.** Approximating arrival time as
   `dep_tod + (30 + Distance/500*60)` minutes, then arrival-hour congestion ratios for
   origin/dest/carrier and 2-hour bins. Delays accumulate through the day; this was the
   single biggest late gain (0.7219 → 0.7233).
4. **Diverse-seed / diverse-hyperparameter ensemble.** 5 → 12 → 18 members (depth 3/4/5,
   colsample 0.6–0.9, subsample 0.7–0.9) plus `colsample_bynode=0.7`. Consistent, robust
   variance reduction: 0.7164 → 0.7229.

## Things that did NOT help (reverted)
- **Raw high-cardinality interactions / target encoding.** `route = Origin_Dest` as a
  categorical (0.7063), target-encoded origin/dest/carrier/route (0.7068), and explicit
  carrier×dow / carrier×month categoricals (0.7046) all overfit 2005 and collapsed on 2006.
- **Explicit time-of-day columns** (`dep_hour`, `dep_min`, `dep_tod`): neutral-to-negative
  (0.7127) — `DepTime` as `hhmm` is already monotone in time, so trees gained nothing.
- **Too many sparse or seasonal aggregates.** route×hour / carrier×route counts (0.7164),
  origin/dest/carrier × month volume ratios (0.7196), carrier+route arrival-hour ratios
  (0.7193), and `max_bin=512` (0.7204). Beyond ~20 features, extra aggregates added noise.

## What I would try with more budget
The dominant constraint is 2005→2006 concept drift: every label-dependent
(target-encoded) or sparse-interaction feature overfits and loses on the held-out year,
so the remaining headroom is in *robust* signal extraction rather than model capacity.
I would pursue (a) a small amount of out-of-fold target encoding computed with a
time-aware CV and heavy shrinkage, evaluated strictly on eval before trusting it;
(b) finer, physically motivated schedule features (per-carrier block time instead of a
single 500 mph constant, turnaround/connection proxies derived from route schedules);
(c) evening out the ensemble with more seeds at a fixed 100 s compute envelope and a
stacked XGBoost meta-learner over member predictions; and (d) validating the
regularization optimum with multiple internal 2005 time folds to distinguish real gains
from the ~0.0005 eval noise floor. Gains below ~0.001 were treated as inconclusive
throughout.
