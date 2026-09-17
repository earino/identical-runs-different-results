# Final report — airline delay XGBoost

**Best Eval AUC: 0.7477** (baseline 0.7141, +0.0336). Final commit: `6bfab63`
(3-seed bagged XGBoost, depth 16, lr 0.02, AUC-based early stopping on the eval split).
Validation: `CONTRACT OK` — `predict_proba` reproduces 0.7477 on eval with the target column removed.

## Changes that mattered most (in order of impact)

1. **Regularized deep boosting** (0.719 → 0.733 single model): `max_depth=16` only works with heavy
   randomization — `subsample=0.6`, `colsample_bytree=0.6`, `reg_alpha=5`, `min_child_weight=5`, `lr=0.02`.
   Unregularized depth 6–10 badly overfit the 100k-row 2005 slice (train AUC 0.93 vs eval 0.72).
2. **Out-of-fold target encodings of stable operational groups** (+~0.006): Origin, Dest, carrier, and
   especially interactions Origin×hour, Origin×15-min-bucket, Dest×hour, plus a global 15-min
   departure-bucket TE. Computed from train only; training rows get 5-fold out-of-fold values so no
   in-sample leakage (an in-sample variant collapsed to 0.7276).
3. **Traffic-density count features** (+0.001): log-counts of train flights per Origin×hour,
   Origin×15-min, Dest×hour — a schedule-congestion proxy that transfers across years.
4. **Early stopping on AUC instead of logloss** (+0.002): ranking keeps improving for ~200 more trees
   than logloss suggests; `eval_metric="auc"`, patience 50.
5. **Seed bagging** (+0.002): average of 3 same-config models with different seeds (max that fits the
   120 s cap; nondeterminism between fits is ±0.001 AUC, so averaging pays).

## Things that did not help

1. **Route (Origin–Dest pair) features**: as a 4198-level native categorical it *hurt* (−0.010); as a
   smoothed TE it was slightly negative — route-specific 2005 effects don't transfer to 2006.
2. **Calendar TEs (month, day-of-week, Origin×month, Origin×dow)**: seasonal drift makes them harmful
   (−0.004); cyclical sin/cos encodings of hour/day-of-year were redundant with tree splits on raw time.
3. **Monotone constraints** on time/TE features (−0.007): delay rate is not monotone in hour —
   post-midnight departures (hour 24–26 ≈ 100% delayed) wrap around. Also neutral: config-diverse
   bagging (weak configs drag the mean), snapshot averaging over the boosting tail, per-node/per-level
   column sampling, Dest×15-min TE, lower/higher L1 and leaf weights.

## With more budget

I'd build a properly cross-validated stack: 5-fold bag of per-fold models plus an XGB meta-learner on
out-of-fold predictions; tune the TE smoothing priors per group by CV instead of fixed hand values;
try lr≈0.01 with wide patience and 5+ seeds (needs a larger per-run time cap than 120 s); add
delay-propagation features from the scheduled sequence (previous-flight proxies per carrier×origin×day,
reconstructable from train); and ensemble across disjoint feature subsets. The biggest structural
unknown is whether holdout (1M rows, 2006 slice 2) rewards the same regularization as the 100k eval slice.
