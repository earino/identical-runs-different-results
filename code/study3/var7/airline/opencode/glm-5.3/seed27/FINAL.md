# Final Report

**Best Eval AUC: 0.7596** (experiment #26, commit 59d2765; baseline was 0.7141)

## Changes that mattered most

1. **OOF target encodings with smoothing** (`te = (sum + k*P0)/(count + k)`, 5-fold OOF values on train
   rows, full-train tables applied at predict time) for high-cardinality keys: origin, route, and the
   time buckets. This took the model from 0.7204 to 0.733+ and is the foundation everything else built on.
2. **Congestion-curve TEs: entity × 15/20-minute departure buckets** — te_t15, te_oh15 (origin×t15),
   te_dh15 (dest×t15), te_ch15 (carrier×t15), and above all **te_r15/te_r20 (route×time bucket)**, which
   alone carries ~72% of gain importance. Time-of-day delay curves per route/airport/carrier are strong,
   year-transferable structure. This was the single biggest win (0.7345 → 0.7519).
3. **Very deep trees** (max_depth 16/18) with strong column subsampling (colsample_bytree 0.30) — deep
   trees exploit the TE interactions; low colsample forces diverse split usage of the TE features
   (0.7521 → 0.7562 from the colsample retune alone).
4. **10-seed ensemble averaging** predict_proba (seed variance on eval is ±0.002; averaging removes it).
5. **Light leaf regularization sweep**: min_child_weight 15/20 → 2/4 (final steps: 0.7590 → 0.7596),
   with count encodings (train-set route/origin/dest/carrier frequencies) retained as supporting features.

## Things that did not help

1. Early stopping on a 2005-internal validation split (picked ~48 trees, badly underfit) and
   early stopping on eval.csv (marginal, fragile); fixed n_estimators=150 at lr=0.02 was better.
2. Alternative time granularities/interactions: t10 buckets, route×10-min and route×12-min TEs,
   month×t15, t15×dow, dest/carrier standalone TEs at shallow depth — all flat or worse than t15/t20.
3. Row subsampling / bagging, monotone constraints, dart, hierarchical (route-prior) shrinkage of the
   r15 TE, and "reliability count" features backing each TE — all neutral or harmful.

## What I would try with more budget

The model is essentially a smoothed route×time congestion estimator, so I'd attack its two remaining
weaknesses: (a) sparsity of route×15-min cells (most have 1–3 training rows) — a properly weighted
hierarchical empirical-Bayes estimate (route×hour shrunk toward route mean, route×15-min shrunk toward
route×hour, with data-dependent shrinkage) or a GAM/boosted component for the smooth time-of-day curve
per airport could replace the crude fixed-k smoothing and likely add a few thousandths; (b) only 2005
data feeds the TEs — recency weighting of the training rows (exponential decay toward late 2005) probed
marginal (+0.0004) but was never combined with the full TE set. I'd also sweep colsample/k jointly with
10-fold OOF, and try 20+ ensemble members with colsample diversity now that runtime allows only 10.

## Budget note

26 of 40 experiments used; stopped when the 18,000 CPU-second budget was nearly exhausted
(~17,900 used incl. final validation) to leave headroom for the hidden-holdout scoring run.
