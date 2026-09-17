# Final report — airline dep_delayed_15min, XGBoost

**Best Eval AUC: 0.7175** (baseline: 0.7141, +0.0034)
Final model: `train.py` @ commit d7798db — depth-4, 280 trees, learning_rate 0.02,
native categorical encoding, baseline feature set (no added features).

## What mattered most (in order)

1. **Shallower trees**: max_depth 6 -> 4 (+0.0010 alone at lr 0.1). On 2005->2006
   time-shifted data, deep interaction trees memorize year-specific patterns;
   depth 4 generalizes better.
2. **Lower learning rate + more trees**: lr 0.1/30 -> lr 0.02/280 (+0.0014 on top).
   Slow shrinkage with a larger ensemble averaged out year-specific noise.
   Curve: 0.1/30: 0.7141 -> 0.1/60: 0.7161 -> 0.05/120: 0.7172 -> 0.02/280: 0.7175.
   Gains flattened below lr 0.02 (0.015/400 tied, 0.02/500 slightly worse).
3. **Tuning tree count to the plateau, not the peak**: AUC was flat 60..280 trees
   at lr 0.1; pushing n_estimators up only helps when paired with lower lr.

## What did not help

1. **DepTime decomposition** (hour/minute/tod/sin/cos), added or replacing raw
   DepTime: 0.7132 vs 0.7141 at baseline params — XGBoost splits raw DepTime
   hierarchically anyway; explicit decomposition diluted the feature budget.
2. **Route = Origin x Dest categorical**: 0.7056 — with 100k rows and ~4600
   routes, per-route stats are too sparse and drift year-to-year.
3. **Early stopping on a 15% holdout** (best_iter=277 at lr 0.1): 0.7070-0.7106 —
   it both discarded 15% of the data and stopped on within-2005 noise, which
   does not track 2006 generalization.

## With more budget

The signal is dominated by a stable time-of-day effect (P(delay) rises from ~0.02
at 5am to ~0.85 around midnight, nearly identical in 2005 and 2006). Next, I would
target features that are stable across years rather than more capacity:
- carrier x hour and origin x hour target/summary encodings computed on 2005 only
  (hour-of-day interactions transfer; route-level ones do not);
- a small ensemble of 3-5 XGB seeds at lr 0.02 averaged (seed noise was ~0.0002,
  averaging may recover half of the last two ticks);
- distance bucket x hour interaction; seasonal (Month) smoothness via cyclic
  month encoding rather than raw c-n levels;
- robustness check: per-carrier AUC on eval to find drifting carriers and down-
  weight or regroup rare ones (F9, RU, etc. had big distribution shifts).

40/40 experiments used, ~6 CPU-minutes of 300 spent, wall clock ~5 min.
