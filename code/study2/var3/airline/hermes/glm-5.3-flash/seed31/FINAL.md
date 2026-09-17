# FINAL — autoresearch XGBoost (airline, scenario 2)

Best Eval AUC: **0.7438** (final `train.py`, `./validate.sh` -> CONTRACT OK, AUC via
`predict_proba` on eval.csv with the target column removed: 0.7438).

Final model: bag of 6 XGBoost models (seeds 42/7/2026 at depth 18, seeds 1/13/99 at depth 16),
all: eta 0.02, 200 rounds, subsample 0.6, colsample_bytree 0.6, min_child_weight 5, lambda 1,
hist, native categoricals; predictions averaged, then clipped to [0, 1].

## Changes that mattered most

1. **Numeric time features with cyclical encodings** (DepTime -> hour/minute, tod/month/dow
   sin-cos, log-distance; string c-<n> calendar fields parsed to numbers): baseline 0.7141 ->
   0.7164, and the foundation for everything after.
2. **Much deeper trees than the defaults suggest**: a depth scan 6 -> 16 with round-count
   checkpoints moved the single-model best from 0.7218 (depth 6 @ 300) to 0.7390
   (depth 16 @ 200). The optimal round count *falls* as depth rises (300 @ d12, 200 @ d16);
   the usual "depth 6-8, many rounds" recipe was leaving ~0.017 AUC on the table.
3. **Row/column subsample retune at the new depth**: 0.7/0.7 beat 0.8/0.8 at depth 8
   (0.7289 -> 0.7296 @ 600 rounds); 0.6/0.6 carried into the depth-16 regime.
4. **Seed + depth bagging**: 3-seed bag 0.7390 -> 0.7421; 6-seed 0.7433; mixed-depth
   (3 x depth18 + 3 x depth16) 0.7438. Pure seed diversity helped most; depth diversity added a hair.
5. **Contract-critical discovery**: xgboost 3.4.1's `Booster.predict` returns raw margins for the
   default squared-error objective (negatives possible). The tuned models are squared-loss
   regressions of the 0/1 label — clipped margins are valid P(Y=1|X) estimates and clipping is
   AUC-invariant, so `predict_proba` clips to [0, 1]. (Training with `binary:logistic` instead
   *lost* 0.004 AUC — squared loss was accidentally the better objective here.)

## Things that did not help

1. **High-cardinality interaction categoricals** — route (Origin>Dest, ~28k levels) plus
   carrier x 2h-block: uniform AUC drop at every round count (0.7079); carrier x block alone
   still lost (0.7196 vs 0.7218). Too many levels for 100k rows without target encoding.
2. **Smoothed target encoding** (m=20 on 9 keys incl. Origin, Dest, hour, origin x month):
   uniformly worse at all checkpoints (0.7117) — 2005-fitted priors do not transfer to 2006.
3. **Early stopping on a time-respecting 2005 probe**: picked 1634 rounds and the transfer
   AUC *collapsed* (0.7048) — in-year validation actively misleads across the year boundary.
   Also inferior: calendar/hour fields as extra categoricals (0.7327), lossguide/128 leaves
   (0.7278), heavier or lighter leaf regularization at depth 16 (0.7374 / 0.7387), lr 0.1 (0.7204).

## With more budget

First, settle the **train-file question**: one run trained on train+eval (eval's labels used
only for fitting) to measure whether adding 2006 data beats the leakage risk on the hidden
holdout. Second, tune **per-member round counts** in the bag (the single-model scans hint the
depth-18 members want ~200 but the depth-16 members ~300). Third, a cheap **year-split CV
inside train.py** (2005-H1 fit / 2005-H2 validate) to tune without touching eval.csv, since
eval-based selection is itself a leakage channel. Fourth, scale the bag to 10-12 members
(~15 s per member) and raise `max_bin` for the fine-grained DepTime splits. Finally, revisit
interaction features *for the deep-tree regime only* — shallow-tree conclusions (route hurts)
may not hold at depth 16-18 where a single split can isolate a route cluster.

## Process notes

40/40 experiments used (~195 min wall clock, ~5.9k of 18k CPU-seconds). Search was strictly
one-change-at-a-time with checkpoint sweeps inside single runs to buy several hyperparameter
probes per experiment; every result equal-or-worse than the best was reverted before the next
idea. Key xgboost 3.4.1 API traps hit: `Booster.evals_result()` removed (use
`early_stopping_rounds` + `best_iteration`), `Booster.predict` returns margins (not probs) for
the default objective, `DMatrix` needs `enable_categorical=True` explicitly.
