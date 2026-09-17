# Final report — airline dep_delayed_15min (XGBoost autoresearch)

Best Eval AUC: **0.7440** (baseline 0.7141, +0.0299). Final config: 7-member
XGBoost ensemble, depths 16..28 (step 2), 150 trees each, learning_rate 0.08,
colsample_bytree 0.4, max_bin 512, hist trees, native categoricals + cyclical
departure-time features. Validated: `CONTRACT OK`, predict_proba reproduces
0.7440 on eval with the target column dropped.

## The 3-5 changes that mattered most

1. **Deep trees + heavy column subsampling (the single biggest lever, ~+0.026).**
   Single max_depth=24, colsample_bytree=0.4 models scored 0.7425-0.7435 on 2006
   data vs 0.7176 for the best shallow config (depth 3-4). Under the 2005->2006
   shift, deep trees restricted to a random 40% of features generalize far better
   than shallow ones; the depth trend kept improving to ~24 and plateaued to 30.
2. **Ensembling decorrelated deep members (~+0.001-0.0015 over the best single).**
   Averaging depths 16..28 with per-member seeds beat any single depth; member
   diversity via depth mattered more than seed diversity (seed-only replicas added
   nothing, depth-spread added the gain).
3. **Cyclical departure-time features (+0.001-0.002).** dep_sin/dep_cos of
   minute-of-day plus dep_hour and dep_minute, computed inside prepare().
4. **max_bin=512 (+0.0005-0.0007).** Finer bins on the numeric time features.
5. **Scaling down per-member trees to fit the 120s cap (150 trees @ lr 0.08).**
   Made the 7-member ensemble feasible within the wall-clock limit at no accuracy
   cost (0.7440 vs 0.7438 for 5x200 trees).

## What did NOT help (all confirmed by experiments or matched probes)

1. **Target-rate (TE) encodings of Origin/Dest/Carrier/hour** — 0.6971 when
   combined with route; even alone they lost to plain categoricals (2005 group
   rates shift by 2006). Route-as-a-single-categorical also hurt (0.7009).
2. **Early stopping on an internal train split** — 0.7114: the random-split val
   AUC (0.755) is unrepresentative of the 2006 shift (0.71), so it stopped too
   early / refit on the wrong signal.
3. **Shallow-tree capacity scaling** (the classic first move): depth 6-8 at
   200-2000 trees always scored below depth 3-4; more trees on deep configs
   (800 @ lr 0.03) also lost to fewer, faster trees (0.7399 vs 0.7435).
4. Seasonal sin/cos, dist-per-minute, hour-as-categorical, daypart blocks,
   frequency encodings, subsample<1, min_child_weight>1, reg_lambda>1,
   rank-average aggregation, lossguide growth, colsample cycling — all neutral
   or worse.

## What I would try with more budget

The dominant pattern is that this task rewards regularized-capacity ensembles
under distribution shift, and the CPU/wall budget (not ideas) was the binding
constraint. With more budget I would: (a) scale the ensemble to 20-40 deep
members spanning depths 16-32 with per-member colsample jitter, since the
7-member average was still ~+0.001 under the single best member and the
diversity curve had not flattened; (b) explore even lower colsample (0.2-0.3)
with more trees per member to compensate, which probed at 0.7293 for depth-spread
but was never tried deep-only with tuned trees; (c) add a second feature axis —
day-of-year aggregates or origin-destination-pair time-of-day interactions —
evaluated with the same train-only discipline; (d) use quantile/rank aggregation
with per-member weights learned on a 2005 holdout month (December) to mimic the
shift; and (e) run a proper nested CV on time-blocked splits (train Jan-Nov,
validate Dec) so hyperparameter choices are made on a shifted split rather than
on eval.csv directly, which should make keep/discard decisions more robust to
the hidden holdout.
