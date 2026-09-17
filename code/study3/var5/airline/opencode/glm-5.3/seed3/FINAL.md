# FINAL.md — final report

## Task
Binary classification of flight delays (target `dep_delayed_15min`). Metric: AUC.
Train = 100k rows of 2005; public eval = 100k rows of 2006 slice 1; hidden holdout = 1M rows
of 2006 slice 2. The year shift is the central difficulty: patterns learned on 2005 partially
fail on 2006 (same-year ceiling measured at 0.7725, so ~0.02 of the residual gap is pure shift).

## Final result: eval AUC = 0.7576 (experiment 19, commit 10745b2, CONTRACT OK)

### Architecture (18 members, prediction-averaged XGBoost ensemble)
`MEMBER_SPECS = [("fullfreq", 4, 0.25), ("cyctime", 4, 0.35), ("cyctime", 4, 0.40), ("sf3r", 3, 0.25), ("sf3r", 3, 0.30)]`
(tuple = feature variant, #seeds, colsample_bytree). All members: `XGBClassifier(n_estimators=100,
max_depth=30, learning_rate=0.05, gamma=1, reg_lambda=2, tree_method="hist", enable_categorical=True)`.
Feature variants (all built inside `prepare()`, all statistics fit on train.csv only):
- **fullfreq** (21 feats): base numerics (month/day/dow, dep time, hour, minute-of-day, distance,
  log distance) + cyclical month sin/cos + day-of-year + 3 interaction categoricals
  (hour×dow, hour×carrier, hour×origin) + log route frequency + log carrier/origin/dest
  frequency + native categoricals.
- **cyctime** (13 feats): base numerics + departure time as sin/cos of minute-of-day + native cats.
- **sf3r** (20 feats): base numerics + dep sin/cos + log frequency of carrier, origin, dest,
  origin×hour, dest×hour, carrier×hour, route×hour (all counted on the 2005 train split) + native cats.

### AUC progression
0.7141 baseline → 0.7222 deep trees → 0.7331 tuned depth/gamma → 0.7387 native categoricals +
interactions + route freq → 0.7450 multi-variant seed ensemble → 0.7513 per-variant colsample
pools + reg_lambda 2 → 0.7536 entity-size frequency features (exp16) → 0.7542 SF/SFR pool split
(exp17) → 0.7552 fullfreq + airport-hour congestion freqs (exp18) → **0.7576 route×hour
congestion freqs (exp19)**.

## The 5 changes that mattered most
1. **Deep, unpruned trees** (max_depth 30, no early stopping, 100 rounds): the single biggest jump
   (+0.011). Depth saturates at ~30; deeper is flat, shallower is much worse.
2. **Native categorical handling + interaction categoricals + route frequency** (+0.05 total over
   the integer-coded baseline). `enable_categorical=True` with `DayOfWeek/Carrier/Origin/Dest`
   passed as pandas Categoricals; hour×entity interactions as extra categorical columns.
3. **Seed ensembling with per-variant colsample pools**: ~4.5–5.5 columns per tree is the sweet
   spot, so each feature variant gets its own colsample (18-feat variant → 0.25, 13-feat → 0.35,
   20-feat → 0.25/0.30). Diversity across both feature sets AND colsample values beat every
   hyperparameter-axis diversity attempt (+0.006).
4. **Entity-size frequency features** (+0.002): log1p of 2005 row counts for carrier, origin and
   dest. Airport/carrier *size* is a congestion proxy that is stable year-over-year — precisely
   the shift-robust feature class this task rewards. All three were needed (super-additive);
   route frequency alone (previously used) is strictly weaker.
5. **Congestion frequency family** (+0.004): origin×hour, dest×hour, carrier×hour and especially
   route×hour log-counts — "how busy is this airport/route at this hour in 2005" — transferred
   remarkably well to 2006.

## The 3 things that did not help (all reverted)
1. **Target encoding / one-hot of high-cardinality categoricals** — one-hot much worse at depth 30
   (0.718–0.730), target encoding flat as a member and dilutive in the ensemble.
2. **Monotone constraints on time-of-day features** — the delay rate is monotone 5:00→24:00 but
   wraps at midnight (0–4h high rates on tiny counts), so global monotonicity costs ~0.015.
3. **Everything-but-the-kitchen-sink ensembling** — HGB members (255-cardinality cap excludes
   Origin/Dest), bagging/subsampling, logit/rank averaging, stacking, weighted blending, extra
   seeds beyond 6 per pool, eta/gamma/max_bin/min_child_weight/alpha sweeps: every one was
   noise-level or worse. Also: week-of-year cyclicals, distance ratios, gamma up-weighting.

## What I would try with more budget
The frequency family is the one direction that kept paying at the end, so I would continue down
that path: third-order congestion counts (route×hour×dow, origin×month), and — more promising —
replace raw log-counts with *supervised* target statistics computed with out-of-fold smoothing
on the 2005 train split (a properly cross-validated target encoding for carrier/origin/dest/
route-hour, which earlier naive TE attempts did not do). Second, attack the year shift directly:
the same-year ceiling is 0.7725, so ~0.015 is pure shift; I would try importance-weighting the
2005 rows by how similar each month's delay marginal is to (a held-out estimate of) 2006, or
dropping the most-shifted months from training. Third, given that wall time (89s of the 120s
budget) and the simplicity criterion both allow it, grow the seed pools of the two strongest
variants (sf3r reached 0.7607 as a standalone 6-seed pool) and test 22–24-member compositions
that were too wall-time-risky at this budget.
