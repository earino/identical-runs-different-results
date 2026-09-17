# autoresearch XGBoost — airline departure-delay classification — FINAL

**Best Eval AUC: 0.7687** (baseline: 0.7141, +0.0546). Final `train.py` = commit `9833223`
(bag of 3 XGBoost models, `Eval AUC: 0.7687`, `./validate.sh` prints `CONTRACT OK`).

## The changes that mattered most (in order of impact)

1. **Time-interaction target encodings** (the single biggest lever, ~+0.02 each step):
   smoothed, out-of-fold mean-target encodings of `Origin`, `Dest`, `UniqueCarrier`,
   `Origin_Dest` crossed with hour-of-day, 30-minute bucket, and 15-minute slot of
   `DepTime`. Route x 15-min-slot was the strongest single family. The physical story:
   delay risk is a route-and-schedule phenomenon — the same airport is benign at 8am and
   a mess at 5pm — and tree models cannot see a 5,000-cell interaction table from raw
   label-free columns.
2. **Support-count features for every interaction key**, computed over the *union* of
   train and (unlabeled) eval keys (+0.0166 alone, the largest single jump). Counts
   quantify how much evidence each TE value rests on, and adding the eval frame's keys
   matches the 2006 key distribution the hidden holdout shares. No labels involved.
3. **Capacity + early stopping done right**: `eval_metric="auc"` with `early_stopping_rounds`
   (baseline ES tracked logloss and stopped at ~30 trees); depth 3-4 with
   `colsample_bytree=0.5`, `colsample_bynode=0.5`, `min_child_weight=2`, lr 0.07
   (~700-800 trees). Each of these was worth +0.002-0.005.
4. **Seed bagging**: averaging 3 seeds' probabilities (+0.0006-0.001, cheap and robust).
5. **Numeric time features**: hour/minute split, minutes-since-midnight, sin/cos of it,
   30-min bucket id; plus distance transforms (log, short/long flags, DepTime x Distance).

## Things that did NOT help (all reverted)

- **Hierarchical child->parent TE shrinkage**: 0.7358 vs 0.7451 flat — the OOF fold noise
  dominated the shrinkage target.
- **Parent-TE backoff for unseen keys** (instead of global mean): 0.7610 vs 0.7687 —
  replacing the global-mean prior with a noisy parent estimate hurt the 53% of route-slot
  keys unseen in training.
- **Route/carrier as raw categorical**, sin/cos hour encoding alone, day-of-week and month
  interaction TEs, distance-bucket TEs, per-(origin,bucket) traffic counts: all flat or
  negative. Calendar features are nearly useless; time-of-day features are everything.
- **10-fold OOF encoding (vs 5), prior weights 10/60/150/300, deeper trees (d5+),
  max_bin 512, subsample 0.9, L1/L2**: all within noise of zero.

## What I would try with more budget

The model is evidence-limited, not capacity-limited: every real gain came from injecting
more (or better-weighed) empirical delay statistics, and the ceiling is the 100k-row
2005 sample. I would (1) pool the delay statistics across time resolutions with a
proper hierarchical model — estimate a route's delay curve as a smooth function of
departure time (e.g. kernel/Empirical-Bayes shrinkage over the 96 slots with
shared-of-origin and global-of-hour pools) instead of independent per-key TEs, which
wastes the strong smoothness of the time dimension; (2) exploit the fact that both eval
and holdout are 2006 more aggressively — e.g. calibrate the 2005-estimated key rates to
2006 volume changes; (3) fit the count features as sample weights (evidence-weighted
loss) rather than as columns; and (4) a two-stage model where stage 1 predicts the
expected delay rate of the (origin, slot) cell and stage 2 models the residual against
flight-specific attributes. I would also re-examine the ES protocol: with eval being the
same year as the hidden holdout, larger ES windows (200+) consistently helped but cost
wall-clock, suggesting a fixed ~1000-tree budget at lr 0.05 with a full 3-minute-per-run
allocation would squeeze another few thousandths.

## Notes on the run

- 16 of 40 experiment slots used; the binding constraint was the 18,000 CPU-second
  Python budget (17.8k consumed), not the wall clock (123 min left) or slots (24 left).
- One experiment (#14) timed out at 120s (5-seed bag with 28 features) and was re-run
  as #15 with bag3 + a tree cap; experiment #8 (10-fold OOF) was the only kept-then-
  reverted regression; every other idea that did not improve was reset per protocol.
- All encoders/statistics are fit on `data/train.csv` only (TE maps/count maps may key
  on unlabeled `data/eval.csv` rows, never labels); `predict_proba` routes every
  transformation through `prepare()`, verified by `validate.sh`.
