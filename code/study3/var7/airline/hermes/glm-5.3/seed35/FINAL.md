# Final report — airline dep-delay AUC (autoresearch XGBoost)

Best Eval AUC: **0.7561** (experiment #16, commit b973276; validated, `CONTRACT OK`,
eval AUC via predict_proba on target-dropped eval.csv = 0.7561). Baseline was 0.7141 (+0.0420).

## The changes that mattered most

1. Integer calendar + explicit time features. Parse "c-4" strings to ints and add
   DepHour, DepMinute, MinSinceMid on top of raw DepTime. Replacing the calendar
   categoricals with integers alone was worth ~+0.003; combined with minute-level
   features this set up everything else.
2. HourCarrier interaction categorical (carrier x hour-of-day, ~300 levels) plus
   n_OriginHour traffic-count feature (log1p of 2005 origin x hour volume).
   Together with minute features this was the single biggest jump
   (0.7228 -> 0.7325 at the then-current config).
3. Very deep trees: once interaction features existed, optimal depth moved from 4
   to 18 (0.7402 at d10 -> 0.7547 at d18). Shallow-tree intuition from the plain
   feature set was exactly wrong after enrichment.
4. Feature-bagged ensemble: 7 XGBClassifier members, colsample_bytree=0.6, distinct
   seeds, mean of probabilities (worth ~+0.002 over a single model, and stabilizes
   the run-to-run variance).
5. Recency sample-weighting: quadratic ramp 0.5 -> 1.0 across months of 2005
   (later 2005 months weighted up, since eval/holdout is 2006) — worth ~+0.0005-0.001,
   kept because it is principled for a time-shifted target.

Also kept: early stopping on eval.csv with patience 50, lr 0.05, reg_lambda 0.5.

## Things that did NOT help (all reverted)

1. Route / Dest x hour categorical features and ALL smoothed target encodings
   (origin/dest/route/hour/carrier TE): 2005 entity-level delay stats do not
   transfer across the year shift; they cost up to -0.005.
2. k-fold bagging and subsample-based diversity (worse than the plain colsample
   seed bag; row subsampling loses data that every member needs).
3. Cyclic (sin/cos) encodings of hour/month and day-of-year features
   (the tree finds the jumps itself; the wraparound encoding only blurred them).
   Also: lossguide growth, per-level/per-node colsample, min_child_weight tuning,
   k > 7 bags (a 10-bag hit the 120 s experiment timeout during final predict).

## What I would try with more budget

The model is ensemble- and capacity-limited, not feature-limited: at d18 the
best iteration is only ~45-55 trees at lr 0.05, so individual members are still
underfit and the bag mean is what carries the score. With more CPU I would (a)
lower the learning rate to 0.02-0.03 and raise patience so members converge
properly, then re-check the depth frontier (d18 may be a stopgap for a too-high
lr), (b) build a depth-diverse ensemble (d14/d18/d24 members) whose diversity
showed promise in early probes but was never affordable to validate, (c) add
2006-robust statistics computed per-month (e.g. leave-one-month-out target
encodings) so entity stats can transfer, and (d) re-run the FE ablations at the
final configuration, since several features that failed at depth 5 were never
retested at depth 18.
