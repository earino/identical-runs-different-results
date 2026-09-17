# Final report — airline dep-delay XGBoost

**Best Eval AUC: 0.7263** (baseline: 0.7141; +0.0122). HEAD = commit `8f941d8`
("add day_of_year numeric feature"), the best-performing kept experiment.
`./validate.sh` prints `CONTRACT OK` and reproduces 0.7263 through `predict_proba`
on a target-free dataframe.

## What mattered most (in order of impact)

1. **Small, heavily regularized models + a diverse bagged ensemble.** With the
   time-separated (2005 train / 2006 eval) split, capacity was poison: the baseline's
   30-tree d6 model beat 200 trees, depth-8 early stopping, and every "big model" we
   tried. The winning structure is 4 configs (d3–d6, lr 0.03–0.05, min_child_weight
   5–20, reg_lambda 3–10, subsample/colsample 0.7–0.85) x 4 seeds = 16 averaged
   XGBoost models (0.7141 -> ~0.7202 including later feature gains).
2. **`day_of_year` numeric feature** (Month c-N + DayofMonth c-N -> a single
   smooth 1..365 seasonal axis): +0.0021, the single largest feature win. Shallow
   trees generalize across adjacent days instead of fragmenting into month/day cells.
3. **Smoothed out-of-fold target encodings conditioned on time-of-day**, most notably
   `dest @ estimated-arrival-hour` (arrival hour = DepTime + Distance/500mph),
   `origin @ (departure hour - 1)`, `carrier @ hour`, plus dow-conditioned variants
   (+0.0025 combined). These encode persistent congestion patterns (late-evening
   arrivals pile up delays) with k=100 shrinkage toward the global mean.
4. **Schedule-density count features** (log counts of flights by origin, dest, route,
   carrier, origin@dep-hour, dest@arr-hour in the 2005 schedule): +0.0010. Pure
   exposure proxies, no label leakage, transfer across years.
5. **Restoring tree count once features got rich**: 100-tree configs were underfitting
   after the TE features landed; doubling n_estimators (100 -> 200/300/450 per config)
   recovered +0.0014. Regularization first, capacity second.

## What did not help (kept out of the final model)

- **Raw capacity on raw features**: 200 trees, depth 8, or early stopping on a random
  2005 validation split (0.7101-0.7119, all below the 30-tree baseline). The year
  shift punishes anything that fits 2005-specific noise.
- **Route as a native categorical** (~5k levels): 0.7103, the worst feature experiment.
  High-cardinality splits are pure 2005 noise at this sample size.
- **Marginal and month-conditioned target encodings** (origin/dest/carrier marginals,
  entity@month), **hierarchical (group->entity-marginal) shrinkage**, and
  **te_hour/te_season target curves**: all neutral to clearly negative. The signal
  lives in time-of-day interactions, not in marginals or seasonal curves trees already
  learn from `day_of_year`.
- **Monotone constraint on DepTime** (0.7225): the true delay-rate curve is not
  monotone (late-night dip), the constraint threw away real structure.
- Micro-tweaks that came out flat and were dropped for simplicity: max_bin=512,
  6 seeds instead of 4, refined arrival-time estimate, cyclic doy sin/cos,
  hour-as-categorical, k=50/k=200 vs k=100 smoothing.

## With more budget

I would (a) tune per-feature smoothing (each TE family probably wants its own k, and a
proper hierarchical prior fit with cross-validated shrinkage rather than the flat
two-level version that failed here); (b) build richer causal congestion features from
the 2005 schedule itself - e.g., airport-level scheduled arrival/departure banks in
30-minute windows around each flight, and "number of scheduled arrivals at dest in the
hour before our estimated arrival" as a density (not target) feature; (c) use
time-aware validation (last months of 2005 as a pseudo-2006) instead of random splits
to make early stopping and config selection track the year shift; (d) stack the four
ensemble configs with a logistic blender fit on out-of-fold 2005 predictions; and (e)
re-test tree count/depth at every feature change, since the under/overfit balance
moved twice during this run.

Budget used: 40/40 experiments, ~179 of 230 minutes, ~3150 of 18000 CPU-seconds.
