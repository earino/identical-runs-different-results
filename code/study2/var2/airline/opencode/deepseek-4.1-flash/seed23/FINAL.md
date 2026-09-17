# Final report — airline delay (AUC)

**Best Eval AUC: 0.7446** (experiment #40, commit `ce80733`)
Baseline: 0.7141. 40/40 experiments used, contract validated (`CONTRACT OK`).

## Changes that mattered most

1. **Deep, heavily-regularized trees.** Overfitting to the 2005→2006 shift was the dominant
   risk. Raising `max_depth` while scaling `min_child_weight`, `reg_lambda`/`reg_alpha` and lowering
   `colsample_bytree` took AUC from 0.7141 to ~0.734 (exps 11–15). Depth 12 was the sweet spot;
   depth 15 with more capacity regressed (exp 16).
2. **Departure-time decomposition + pruning.** Splitting `DepTime` (hhmm) into hour/minute/minutes,
   cyclical sin/cos and `log_distance`, while dropping the redundant raw `DepTime`, `Distance` and
   the noisy `DayofMonth`, gave a clean +0.005 early on (exps 5, 10).
3. **Target-free congestion / frequency features fitted on train only.** The largest single lever.
   Origin/Dest/Carrier-at-hour frequencies (exp 26), then their normalized shares (exp 28),
   hour-relative concentration (exp 30), observed/expected *lift* (exp 31), and departure-vs-arrival
   differences (exp 35) compounded to ~0.744. Route frequency (exp 25) and route competition
   (exp 37) added smaller gains.
4. **XGBoost ensemble.** Averaging 4 models at depths 10–13 with different seeds (exps 19, 33)
   removed single-model variance and edged out the best single model.
5. **Categorical hour** alongside the numeric hour gave a small consistent gain (exp 34).

## Things that did not help

- **High-cardinality identity features** — `route = Origin_Dest` as a category (exps 6, 18) and
  smoothed **target encoding** of airports/carrier (exp 8) both hurt badly; they memorize
  2005-specific patterns that do not transfer to 2006. Their *frequency* counterparts generalized
  instead.
- **Explicit time-interaction integers** (`dow_hour`, `month_hour`) and cyclical month/dow features
  (exps 20, 21) — deep trees already capture these, and the raw interactions just add overfitting room.
- **Bigger/less-focused ensembles and deeper-than-12 trees** — a 5-model diverse ensemble (exp 23,
  0.7335) and depths 11–14/14 (exps 24, 36) were worse than the compact 4-model depth-10–13 ensemble.
- Micro-tuning learning rate to 0.02 with 2000 trees (exp 17) matched but did not beat the simpler
  1000-tree/lr-0.03 config, so it was discarded on simplicity grounds.

## What I would try with more budget

The winning theme was *target-free, train-fitted density/competition statistics*, so I would push
further in that family rather than tune hyperparameters: finer time buckets (30-minute banks instead
of hours), route-level carrier-share ranks and airport market concentration (HHI), and explicit
scheduled-bank structure (arrivals vs departures per airport-hour) — all computed from training data
only and applied inside `prepare()`. I would also try probability calibration or rank-averaging the
ensemble (AUC-only, but rank averaging can help when members are correlated), and a proper temporal
validation split (hold out late-2005 months by `Month`) to make keep/discard decisions less dependent
on the single 2006-slice eval set. Finally, a small amount of `num_parallel_tree`/`colsample_bynode`
randomization within XGBoost could add ensemble diversity without the cost of extra full models.
