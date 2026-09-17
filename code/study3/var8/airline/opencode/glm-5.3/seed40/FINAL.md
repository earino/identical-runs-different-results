# Final report — airline dep-delay AUC benchmark

**Best Eval AUC: 0.7405** (experiment #33, commit `226ac0b`), up from the 0.7141 baseline.
Final `train.py`: 20-model XGBoost ensemble (2 seeds x 10-fold CV, per-fold early stopping),
`max_depth=13`, `colsample_bytree=0.4`, lr=0.07, with deconstructed DepTime/calendar features and
three smoothed target encodings. Validation: `CONTRACT OK`.

## Changes that mattered most
1. **Random-forest-style feature bagging + deep trees inside a boosted ensemble**
   (`colsample_bytree=0.4`, `max_depth=10→13`): the single biggest lever. Offline sweeps showed
   cs0.4+d16..24 reaching 0.7428–0.7447 at 20 models; under the 120 s/experiment cap the best
   reproducible point was d13 → 0.7397 → 0.7405 (vs 0.7227 before this direction).
2. **10-fold CV ensembling with per-fold early stopping** (+0.005): average 20 models trained on
   90 % row subsets; ES per fold picks ~200–400 trees adaptively. 5→10 folds and seed bagging
   (2 seeds) each added a little; a full-data model added early was later dropped (in-sample TE).
3. **Smoothed target encodings fit on train only**: `te_Hour` (w=200), `te_OriginHour`,
   `te_DestHour` (origin/dest x 3h-bucket, w=50) — +0.0013. These encode stable
   "airport-by-time-of-day congestion" that transfers across the 2005→2006 shift.
4. **DepTime deconstruction**: hhmm integer → hour, minute, minutes-since-midnight (+cyclic sin/cos)
   and c-N calendar strings → integer + sin/cos features (baseline boost, foundational for the TEs).
5. **Feature-set pruning for the bagged regime**: dropping raw string categoricals
   (Month/DayofMonth/DayOfWeek) and raw DepTime so colsample draws only informative features —
   the aligned feature set was necessary for the d10–d13 gains to transfer officially.

## Things that did NOT help (all reverted)
1. High-cardinality interaction categoricals (Origin-Dest route pair 4198 levels, carrier x hour):
   0.7085 vs 0.7145 — year-shift overfitting.
2. Fine-grained target encodings (origin/dest x 1-hour, route x 3h — 4k–17k levels): 0.6878, a
   catastrophic memorization of 2005 idiosyncrasies; plain route/origin/dest/carrier TEs also hurt.
3. Regularization/structural variants on the shallow config: min_child_weight 10–20,
   subsample 0.8, reg_lambda 2, max_bin 512, lossguide trees, day-grouped CV folds, lr 0.05 with
   depth 7 — all neutral to negative (offline probes were all flat 0.7217–0.7223).

## With more budget
The offline probes showed a clear runway that the 120 s per-experiment cap blocked:
colsample 0.3–0.45 with `max_depth=16–24` reached 0.7428–0.7447 at 20 models, and deeper
(24) at 0.7447 — each such run needs 2–4 minutes. I would (a) move the ensemble to d16–d24 with
more models and a higher wall-clock allowance (the per-model cost is the binding constraint, so
more CPU threads or a relaxed timeout directly buys AUC), (b) tune colsample/depth jointly around
(0.35, d18) since the surface was still rising at the cap, (c) replace in-sample TE with
out-of-fold TE per fold to clean up early-stopping selection, and (d) add max_bin=512 plus
carrier-level coarse TEs that were skipped for CPU reasons. A proper time-shifted validation
scheme built from the 2005 calendar would also make keep/discard decisions safer for the hidden
holdout than eval.csv alone.
