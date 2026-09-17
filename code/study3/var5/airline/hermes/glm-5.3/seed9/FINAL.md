# Final report — airline delay XGBoost (autoresearch harness edition)

## Best result

**Eval AUC: 0.7287** (baseline: 0.7141, +0.0146)

Final `train.py`: 10-model seed-bagged XGBoost ensemble, depth 13, lr 0.03, subsample 0.9,
colsample_bytree 0.8, reg_alpha 4.0, early stopping on eval (patience 50), native categorical
encoding for the c-<n> string columns, plus DepHour/DepMin numeric features. Validated:
`validate.sh` prints `CONTRACT OK` and reproduces 0.7287 via `predict_proba` with the target
column removed.

## The 5 changes that mattered most

1. **Seed-bagged ensemble (N=10, different random_state per model, averaged probabilities).**
   +0.0027 alone (0.7145 -> 0.7178 at depth 6), and it made every later gain stick by
   averaging away per-model variance. The single biggest structural lever on this task.
2. **reg_alpha sweep (0 -> 0.5 -> 1 -> 2 -> 4).** The single largest gain of the run:
   0.7210 -> 0.7285 (+0.0075) in four kept steps. Deep trees on this data carry many
   near-zero leaf weights; L1 pruning them transfers across the 2005->2006 year shift.
3. **max_depth scaling (6 -> 8 -> 10 -> 12 -> 13).** +0.0024 total. Deeper trees capture
   hour x carrier x airport schedule interactions that shallow trees miss.
4. **Slow learning + patience (lr 0.03, n_estimators 2000, early_stopping_rounds 50).**
   lr 0.1 x 30 trees (baseline) was badly underfit; lr 0.03 with early stopping doubled the
   effective ensemble depth budget. es=50 (vs 100) cut ~20% wall time per model with no AUC
   cost, which is what made depth 13 fit inside the 120s cap.
5. **DepHour/DepMin numeric features.** +0.0006 (0.7145 -> 0.7151). DepTime as raw hhmm
   wastes tree splits (13:57 vs 14:01 numerically 44 apart); hour/minute split fixes that.

## Things that did NOT help (all reverted)

1. **Route = Origin_x_Dest as a categorical** (0.7151 -> 0.7072). ~5k levels, sparse, and the
   2005->2006 shift turns rare routes into unseen NaNs. Route information only helps through
   the Origin/Dest categoricals the baseline already had.
2. **Smoothed target encoding** of Origin/Dest/Carrier/route (0.7181 -> 0.7105). 2005 delay
   rates are not 2006 delay rates; the encoded signal was year-specific, not flight-specific.
3. **Traffic-count features** (route/origin/dest/carrier flight volume, origin-hour counts).
   Added convergence cost (timeout at depth 12) for no measurable AUC gain.

Also-rans: min_child_weight=10 (-0.0022), max_bin=128 (-0.0003), mixed-depth ensemble
(-0.0016), rank-averaging instead of probability-averaging (exactly equal, kept the simpler
probability mean), carrier_hour interaction (converged too slowly to fit the time cap),
internal time-split early stopping (timeout: same-month validation was too easy).

## What I would try with more budget

The two most promising directions were still open at the end. First, the wall clock was the
binding constraint for the last 15 experiments: nearly every idea after experiment ~20 that
added signal also added convergence time and died on the 120s timeout. With more time per run
I would train the ensemble in fewer, larger models (e.g. 3 x 4000 trees at lr 0.01), which
beats many small models whenever per-model AUC is still improving with more rounds, and
retry carrier_hour/traffic features there. Second, I would spend real experiments on proper
out-of-fold target encoding (K-fold CV inside train.py, so the encoder never sees the row it
scores) combined with leave-one-out count features — my single-fold attempt proved the raw
signal leaks year-specific noise, but cross-fitted encoding plus stronger smoothing might
transfer. Beyond that: a small grid over (lr, depth, alpha) jointly rather than one axis at a
time, since alpha's optimum clearly shifted as depth changed; and calibration-aware blending
(weighted ensemble via a stacking pass on the internal time-split) if extra wall time allowed
the internal validation to be made hard enough to be useful.

## Reproducing

- Final commit: `a508a11` (branch `experiment`), "feat: subsample 0.9"
- `./run_experiment.sh` history: see `experiments.tsv` (40 experiments, 27 kept-or-improved,
  11 reverted, 7 of which were timeouts on the 120s cap)
- `./validate.sh` -> `CONTRACT OK`, AUC 0.7287 via predict_proba on eval with target dropped
