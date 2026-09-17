# autoresearch XGBoost — airline delay (final report)

**Best Eval AUC: 0.7383** (commit `5f45739`, experiment #34 of 40; baseline 0.7141 → +0.0242).
Contract check: `./validate.sh` prints `CONTRACT OK`; eval AUC recomputed through `predict_proba(df)` with the
target column removed is identical (0.7383), so all feature engineering is reproducible on unseen rows.

Budget used: 40/40 experiments, ~123 minutes of the 230 allowed, 7,220 of 18,000 CPU-seconds.

## The 5 changes that mattered most

1. **Out-of-fold target encoding of the categorical keys (+0.0095 in one step, experiment #5).** Route, origin,
   dest, carrier, hour, month, day-of-week and their interactions, smoothed toward the prior with m=20, computed
   out-of-fold (5-fold KFold) for the training rows and from the full train slice at inference. This replaced the
   raw categorical features as the main carrier of signal.
2. **Dropping the high-cardinality raw categoricals (Origin/Dest/route as `enable_categorical` features, +0.0077,
   experiment #16).** Their partition-based splits fit the 2005 slice far better than 2006 — the single biggest
   single-knob gain in the whole run. Low-cardinality categoricals re-added later also hurt (#23, 0.7336), so the
   model now sees only numeric/time features plus encodings.
3. **Ensembling: a hyperparameter-diverse bag of XGBoost models (3 → 6 → 12 models, +0.0035 total).** Members
   vary depth (5/6/7) and `colsample_bytree` (0.6–1.0). Variance reduction is the most reliable lever here — it
   helped at every size, and it should transfer to the 1M-row hidden holdout.
4. **Count/volume features alongside every encoding (+0.0034 when removed, experiment #28).** `log1p` frequency
   per key lets the trees discount noisy encodings for rare keys; they are not decoration.
5. **Network-structure features fitted on train only (+0.0006, experiment #34).** Airport size (distinct
   destinations), route carrier count, carrier network breadth and log market shares — unsupervised, so they
   transfer across the year boundary. Also, a longer/slower boost (`2000 trees, lr 0.01`) plus `subsample=0.7`
   gave small, consistent gains once the features were smooth.

Modelling: `XGBClassifier(hist)`, 6 models, 2000 trees each, `lr=0.01`, `max_depth` 5–7, `min_child_weight=10`,
`subsample=0.7`, `colsample_bytree` 0.6–1.0, `reg_lambda=2`, predictions averaged. Runtime ~83s, well under the
120s cap, ~2.5GB peak.

## The 3 things that did not help

1. **More capacity on raw features.** 300 trees at lr 0.05 (0.7119) and 600 at 0.04 (0.7227) both *lost* to the
   30-tree baseline (0.7141) while the raw categoricals were in the model; depth 8 lost too (0.7231). The
   2005→2006 gap, not underfitting, was the binding constraint.
2. **Seasonal/calendar encodings.** route×month, carrier×month, dest×month and day-of-month encodings lost in both
   regimes (0.7247 and 0.7326); the raw `Month`/`DayofMonth` categoricals also hurt. 2005's calendar effects do
   not transfer to 2006.
3. **Refinements of the encoding itself.** Sharp (m=3) second encodings (0.7267), 10-fold instead of 5-fold OOF
   (0.7343), hierarchical backoff to parent keys (0.7368), SMOOTH=50 (0.7360) and TE "delta/premium" features
   (0.7360) were all neutral or slightly negative. So were finer time-of-day/dist-bin keys, `min_child_weight=20`,
   and extra origin/dest×hour×dow keys in the final stretch — all within ±0.0007 of the best.

## What I would try with more budget

The forward gap (train-holdout 0.745 vs eval 0.712 in the early diagnostic, narrowing to ~0.74/0.738 later) says
remaining gains come from domain-shift robustness, not model capacity. I would (a) fit the interval estimates of
each key's rate with a proper hierarchical/Bayesian model and feed the *posterior uncertainty* alongside the mean,
letting the trees trust stable keys more than volatile ones; (b) build row-level "hub bank" features from
origin×carrier×hour structure and, more importantly, (c) exploit the fact that eval and the hidden holdout are
*time slices* — a model trained only on the stable part of the 2005 slice (routes/carriers present across many
months) and ensembled with the current all-data model may transfer better; (d) run a proper 5-fold-CV hyperparameter
search with the CV score as the selection metric rather than eval AUC, since at ±0.0015 noise the last five
experiments were mostly coin flips; and (e) with much more compute, an ensemble of 20–30 diverse members with
different encodings (count-based, rate-based, raw-categorical) would likely add a few tenths of a point of AUC
without overfitting the 2006 slice.
