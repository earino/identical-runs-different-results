# Final report — airline dep_delayed_15min

**Best Eval AUC: 0.7474** (experiment #40, commit 0b6c237; baseline was 0.7141, +0.0333)
Final train.py validated: `./validate.sh` prints `CONTRACT OK` and reproduces 0.7474 through
`predict_proba(df)` with the target column removed.

## Changes that mattered most

1. **Deep trees (max_depth 8 → 30).** The single biggest lever, +0.016 AUC. Deep trees discover the
   route × carrier × time-of-day interactions internally, which is exactly why every hand-built
   interaction feature failed. AUC kept improving monotonically from depth 8 through 30 and only
   flattened around 40.
2. **Early stopping on AUC instead of the default logloss.** +0.003 on its own, and it unlocked
   everything else: under the 2005→2006 calibration shift, logloss stops improving while the AUC
   ranking is still gaining, so logloss-based stopping cut training at ~40 rounds vs ~275 for AUC.
3. **min_child_weight 1 (from 5).** +0.005. Deep trees want fine leaves; coarser leaf constraints
   (mcw 20) were disastrous (−0.009), looser ones monotonically better.
4. **Ensemble of deep XGBoost models** (seeds + depth 24/30/36 + colsample 0.7/0.8 diversity,
   probability-averaged). +0.0023, and it makes the final model more robust to the shifted holdout
   than any single member.
5. **Low learning rate (0.012) with ~300 rounds and max_bin=512.** Finer shrinkage and finer
   DepTime histogram resolution each contributed a few ten-thousandths; together with the above
   they define the final configuration.

## Things that did not help

1. **Hand-built interaction categoricals** (Origin_Dest route, Carrier×Origin, Origin×Hour as raw
   categorical strings): 0.7026 at depth 8 (−0.017) — they memorize 2005-specific combinations —
   and at depth 30 the ~5000-level OriginHour/DestHour columns slowed training into the timeout.
2. **Smoothed target encoding** of the high-cardinality columns (Origin, Dest, Route, Carrier×airport,
   Origin×hour): 0.7125, worse than just letting XGBoost partition native categoricals.
3. **Bagging the ensemble members on 90% row subsets**: 0.7454 vs 0.7470 — losing 10% of the data
   cost more than the added decorrelation gained. (Likewise subsample/colsample 1.0 crashed to
   0.7268: row sampling is essential regularization for deep trees, but it should happen inside
   boosting, not between members.)

## What I would try with more budget

The 120-second per-run cap forced the ensemble down to 2–3 members; with a looser cap (or faster
hardware) I would first scale the ensemble to 6–10 members with wider diversity (depth, colsample,
learning rate, maybe `max_leaves`/`loss_guide` growth), which historically keeps paying +0.001–0.003.
Second, I would add scheduled-congestion features computed from the training slice only — flights
per (Origin, hour) and per (Origin, day) — since airport-level queuing is the physical driver of
departure delay and is not directly observable from a single row. Third, I would replace eval-based
early stopping with k-fold out-of-fold AUC on the training split for the keep/discard decisions,
reserving eval for a final sanity check, to reduce the risk of overfitting 2006 slice 1 — the same
overfitting risk that made me distrust sub-0.0005 gains. Finally, I would test a combined refit of
the winning configuration on train+eval (2005 + 2006 slice 1) so the deployed model has seen the
2006 distribution, checking on a held-out 2006 slice whether the extra 100k labeled rows help or
hurt the time-shifted holdout.

## Budget usage

40/40 experiments, ~52 of 230 wall-clock minutes, ~8,550 of 18,000 CPU-seconds.
Experiment log: `experiments.tsv`. All keep/discard decisions follow program.md (revert on
non-improvement, keep on improvement).
