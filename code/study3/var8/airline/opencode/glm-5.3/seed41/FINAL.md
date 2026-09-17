# FINAL — airline dep-delay, XGBoost benchmark edition

**Best Eval AUC: 0.7599** (baseline 0.7141, +0.046 over 40 experiments; HEAD = best commit `a2b592b`).

## Changes that mattered most (in order of impact)

1. **Heavy bagging**: `subsample=0.5, colsample_bytree=0.5` — the single biggest jump (+0.012 alone, 0.7284→0.7405). Stochastic regularization is essential on this noisy, 100k-row dataset.
2. **Very deep trees + low learning rate + early stopping on eval**: depth ramp 6→14→24 and lr 0.1→0.007 with `early_stopping_rounds=250` (best_iter≈300). Deep trees capture hour×carrier×airport interactions; the 120s cap made lr<0.006 impractical.
3. **Busyness/frequency features** (fit on train only): route/origin/dest/carrier counts (+0.0044), plus relative-distance-to-route-mean and origin/dest-hour traffic counts as congestion proxies (+0.0008 more).
4. **Time-of-day engineering**: hour/minute from DepTime + cyclic sin/cos of minutes-since-midnight; 2nd and 3rd harmonics added +0.0011 combined; categorical hour added +0.0005. Dropping the cyclic features cost -0.005, confirming their value.
5. Early stopping on eval.csv (2006, same year as the hidden holdout) — safe model selection that transfers across slices.

## Things that did NOT help

1. **Origin-Dest route pair as a native high-cardinality categorical** (-0.02): sparse 2000-level category splits wasted tree capacity; XGBoost's per-node one-hot partitioning was destructive at depth 24.
2. **OOF target encoding of route** (-0.0015): trees already extract route signal from Origin/Dest/freq features.
3. **Seed ensembles** (2-seed average: slightly worse / timeout — each deep model is too slow to fit twice in 120s), **lossguide growth** (-0.011), **reg_lambda/min_child_weight/colsample_bynode/max_bin tweaks** (all ≤0), **cyclic minute-of-week** (-0.004).

## With more budget I would try

Fit the time constraint head-on: distill the winning config into a faster one (depth ~16, lr 0.02, es 100) and average 5-10 seeds as a proper bagged ensemble, which typically beats any single model by +0.001-0.002 on 1M-row holdouts; explore smoothed delay-rate features at the (origin, hour) and (carrier, hour) level with honest OOF construction; run a small random search over (subsample, colsample, depth, min_child_weight) around the current optimum with 5x2 CV on time-blocked folds of train (never touching eval for selection) to guard against eval-selection noise; and test whether adding the 4th/5th time harmonic or piecewise-linear hour basis continues the harmonic trend. CPU budget was never binding (7.1k/18k CPU-s used) — more experiments or bigger ensembles were limited by the 120s per-run wall clock, so a two-process parallel ensemble would be my first move.
