# Final report — airline delay XGBoost (autoresearch)

**Best Eval AUC: 0.7521** (experiment #39, commit `edaf24b`), vs. baseline 0.7141 (+0.0380).
Final model: 7-fold bagged `XGBClassifier` ensemble, `max_depth=24`, `learning_rate=0.03`,
`subsample=0.8`, `colsample_bytree=0.5`, `min_child_weight=5`, `reg_lambda=1.0`, early stopping (50) on
each held-out fold; probabilities averaged across the 7 fold models. Contract validated (`CONTRACT OK`).

## Changes that mattered most
1. **Time-of-day parsing.** `DepTime` (hhmm) → continuous minutes-since-midnight plus hour/minute and
   cyclical `sin/cos`. The delay-vs-hour curve is the single strongest and most year-stable signal
   (train↔eval correlation r=0.96); parsing fixed the artificial jumps at hour boundaries.
2. **Removing calendar features (`Month`, `DayofMonth`).** Their delay effects do **not** transfer from
   2005 to 2006 (month r=0.59, day-of-month worse). Ablating them gained ~+0.006 AUC. `DayOfWeek`
   (r=0.96) was kept.
3. **Rare-airport bucketization + unseen-category routing.** Orig/Dest with <300 training flights are
   collapsed into a shared `__RARE__` level, and *unseen* eval/holdout airports are mapped to that same
   level instead of becoming missing values. ~14% of rows move to `__RARE__`; added ~+0.003.
4. **Airport traffic frequency.** `log1p` counts of departures per `Origin`/`Dest` (fit on train only)
   as extra numeric features; added ~+0.002.
5. **Deep trees + K-fold bagging + strong feature subsampling.** Depth mattered a lot (depth 7→24 gave
   ~+0.015); averaging 7 fold-models stabilizes it. `colsample_bytree=0.5` was the best of a sweep
   (0.8→0.5 improved monotonically, 0.4 regressed).

## What did not help
1. **Categorical/target encodings of route, origin, dest, carrier** (native route categorical, OOF
   target encoding, frequency-conditional encodings). Airport delay *rates* and route-level patterns
   are strongly year-unstable (r≈0.26–0.38) and consistently hurt, even with OOF/smoothing.
2. **Stronger regularization** (`min_child_weight` 5→20, `reg_lambda` 1→5, `subsample`/`colsample`
   0.6–0.7): large drops — this problem rewards capacity, not shrinkage.
3. **A fixed holiday-window flag** (Thanksgiving/Christmas/July-4 windows). Real effect in isolation
   (0.576 vs 0.493 delay rate, consistent across years) but no eval gain, presumably because it is too
   coarse without the surrounding calendar features.
4. **8-fold ensemble**: exceeded the 120 s experiment limit (timeout); 7 folds is the practical max.

## What I would try with more budget
The 2005→2006 gap (in-domain fold AUC ≈0.80 vs. cross-year 0.75) is the dominant limiter; adversarial
validation shows the shift is driven by carrier composition, then airport identity. With more time I
would (a) build carrier- and airport-conditioned *hour-profile* features and encode them relative to
the stable global hour curve rather than as raw target rates, (b) add monotone constraints / smooth
splines on departure time, (c) increase ensemble diversity (mixed depths and seeds, 2-level stacking)
at the same compute, and (d) search rare-bucket thresholds jointly with depth. Most importantly I would
tune and select on a *year-split* internal validation (e.g. 2005 months held out) rather than the random
folds, to make model selection itself robust to the temporal shift.
