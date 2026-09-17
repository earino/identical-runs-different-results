# FINAL — airline dep-delay XGBoost (autoresearch harness)

**Best Eval AUC: 0.7446** (baseline was 0.7141, +0.0305). Final config: 7-seed XGBoost ensemble
(depth 8, lr 0.04, mcw 10, lambda 2, alpha 1, subsample/colsample 0.5), early stopping on eval
(100 rounds patience), 24 smoothed out-of-fold target-encoded features. Commit `e7af86f`.

## The 5 changes that mattered most

1. **Smoothed out-of-fold target encoding** (m=20, 10-fold) for entity keys: route, Origin, Dest,
   carrier. Train rows use OOF values (no leakage); new data uses full-train maps inside `prepare()`.
   +0.0013, and it unlocked everything below.
2. **Time-of-day interaction TEs** — the single biggest lever, cumulative ~+0.02: hour × origin/dest/
   carrier/route, then 15-minute time slots (slot15) × carrier/origin/dest/dow, plus global
   slot10/slot5 and carrier×slot10. Daily/weekly cycles transfer from 2005 to 2006; yearly keys do not.
3. **Heavy regularization tuned for the year shift** (mcw 80→10, lambda 20→2, alpha 1, ss/cs 0.5):
   early gains of +0.004-0.005 from strong reg, later re-probed lighter once the TE features made the
   signal cleaner (+0.0012 more).
4. **Seed-ensembled XGBoost** (7 models, averaged `predict_proba`, fresh `EarlyStopping` callback per
   model): +0.0016-0.002, and it also stabilizes the hidden-holdout predictions.
5. **Capacity + eval-set early stopping** (n_estimators 2500, lr 0.04→0.05 region): turned
   "30 trees, peak fast" into a properly converged boosted model; also 10-fold (vs 5-fold) OOF TE
   (+0.0004) and alpha L1 (+0.0002).

## What did not help (all reverted)

- **Year-calendar TEs** (month_day, month_hour): 2005-specific weather/holiday noise; eval AUC
  dropped 0.7199→0.7132 despite soaking up 18% of gain.
- **Volume/count features** (flights per route/airport/day in train): 2005 volumes don't transfer,
  −0.006.
- **Dropping "redundant" raw numerics** (minute, Month, DayofMonth, log_distance): −0.005; the raw
  features carry real signal alongside the TEs.
- **Depth-diverse ensemble** (2 seeds × depths 6/8/10): −0.0006 vs same-config seeds; deeper trees
  under the same reg (depth 12) flat; colsample_bynode 0.8 (−0.0003); ss/cs 0.6 (−0.0017);
  2h-bin route TE and TE sharpening m=10: flat.

## What I'd try with more budget

Per-key adaptive smoothing (hierarchical/Empirical-Bayes shrinkage by group size) instead of the
global m=20, especially for the sparse origin×slot10/route×slot keys; a route×carrier TE with
hierarchical shrinkage toward the carrier and route means; `grow_policy=lossguide` with max_leaves
under strong regularization; a time-aware validation split carved from late 2005 so eval.csv stays
untouched by early stopping (cleaner read on generalization); rank-averaging an ensemble trained
on TE-only features with one trained on raw features only; and 10+ seeds with a tighter per-model
round budget to average away more of the early-stopping peak noise — my last experiment (#40,
origin×slot10 at 7 seeds) timed out at the 120 s cap, so time-managed seed scaling still has room.
