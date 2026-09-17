# FINAL — airline dep-delay XGBoost (autoresearch)

**Best Eval AUC: 0.8282** (baseline 0.7141, +0.1141). 40 experiments used, 0 crashed at the end.
Final config: commit `c578125` — 59 engineered features, XGB (d=5, mcw=100, lr=0.03, subsample=0.9,
colsample=0.7, early stopping on 10-fold OOF folds = cvbag10 ensemble), all engineering inside `prepare(df)`.

## Changes that mattered most
1. **Within-df congestion/queue counts** (label-free, computed from the passed df, scaled by 1000/len(df)):
   rolling ±15min and backward windows (30/60/90/120/180/240min) over Origin, Dest, and Origin×Dest route
   keys — the single biggest family (0.727 → 0.80).
2. **Day-of-queue position + fractions**: same-key flight counts earlier in the calendar day, plus
   DayFwd and position fractions (0.817 → 0.821).
3. **Spacing gaps**: normalized minutes since previous/next same-route/airport flight (0.817 → 0.824).
4. **Carrier×airport queue**: same-carrier-at-same-airport backward counts (carrier cascade effect,
   last structural win: 0.8261 → 0.8279).
5. **CarrierHour categorical + cvbag ensembling + shallow/regularized trees** (d4-5, mcw 50-100,
   subsample/colsample 0.7-0.9): robustness to 2005→2006 drift; 10-fold bag reliably +0.001-0.002.

## Things that did not help
1. Target encodings of any kind (all variants hurt — drift-sensitive).
2. High-cardinality interaction cats (Route concat, Origin/Dest×Carrier, Carrier×Block/Dow) — all hurt.
3. Deeper trees (d6/d7), lr 0.02, colsample 0.8-0.9, subsample 0.6/1.0, reg_lambda=2, esr 50/80,
   full-data refit, depth-diverse ensembles — all tied or lost vs the chosen config.

## With more budget
The winning direction was physical queueing structure, which is still not exhausted: per-flight
"wake" features (congestion a flight *causes* for its downstream schedule), queue features conditioned
on day-of-week and month (demand seasonality interacts with congestion), time-of-day-normalized
queues (a 30-min backlog means something different at 6am vs 6pm), and two-stage carrier cascade
propagation (carrier's previous arrival feeding its next departure). I would also validate
congestion-feature transfer on a denser slice (the 1m holdout averages ~10× denser queues; the
1000/len scaling assumes uniform sampling) and probe a light two-model ensemble (shallow drift-proof
+ deep signal model) with more wall-clock per run.

— autonomous run, 40/40 experiments, ~30 min wall remaining at finalize.
