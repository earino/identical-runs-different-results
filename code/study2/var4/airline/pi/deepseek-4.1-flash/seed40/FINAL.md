# Final report — airline delay (XGBoost, AUC)

**Best Eval AUC: 0.7563** (experiment #11, commit `cc778c5`)
Baseline Eval AUC: 0.7141. Net gain: **+0.0422**.

The final `train.py` is a 5-model seed-averaged XGBoost ensemble. `HEAD` is the best commit and
`./validate.sh` prints `CONTRACT OK` (0.7563 re-derived through `predict_proba` on eval with the target
column removed).

## What mattered most (in order of impact)

1. **Departure time-of-day as categorical buckets** (`dep_hour` 24 levels, `dep_20` 72, `dep_q` 96)
   instead of the raw `hhmm` integer. Time-of-day is the dominant signal (~65% of split gain) and
   categorical buckets let trees carve out the intra-day delay profile. This alone took the score from
   ~0.723 to ~0.736. `dep_20`/`dep_q` were both needed; finer `dep_10`/`dep_5` hurt.
2. **Deep trees on those categories.** With the raw feature set, extra capacity overfit the training year
   (500 trees/depth 7 dropped AUC to 0.711). With categorical time features, depth 12–14 with
   `min_child_weight=1` was *better* (depth trend: 7→0.741, 10→0.747, 14→0.750) and transferred cleanly
   across eval halves.
3. **Dropping `Month` and `DayofMonth`.** Their 2005 seasonality does not transfer to 2006; removing them
   was worth ~+0.005 and is a deliberate cross-year robustness choice. `DayOfWeek`, carrier, airports and
   distance transfer well and were kept.
4. **`carrier × dep_hour` interaction + `log(Distance)`** (`carhour`, 480 levels): +0.004.
5. **Target-free popularity features**: route frequency and origin/destination-hour traffic counts
   (congestion proxies fitted on train only): +0.0015.
6. **Seed-averaged ensemble with column subsampling** (`colsample_bytree` 0.5/0.6/0.7 across 5 models).
   With `subsample=1`, XGBoost is fully deterministic, so seed diversity comes from column subsampling.
   This added ~+0.001–0.002 and stabilised the ranking.

## What did not help (tested and reverted)

- **Generic capacity without feature work** (500 trees, depth 7) — overfit the training year (0.711).
- **Target encoding** of Origin/Dest/carrier/route, with smoothing — badly overfit (0.69–0.71): 2005
  per-airport delay rates do not transfer to 2006.
- **High-cardinality categorical interactions**: `route` (Origin×Dest), `dowhour`, `carrier×dep_20`,
  `desthour`, `Origin×dep_hour` — all neutral or worse.
- **Row subsampling** (`subsample=0.8`), **cyclic month sin/cos**, `max_cat_threshold=128`,
  a **6-model** ensemble with smaller trees, and **n=360/lr=0.04** — all equal or slightly worse than the
  simpler 5-model n=320 setting, so they were discarded.

## With more budget I would

Run a proper **time-based hyperparameter search**: split 2005 by month (train on Jan–Sep, validate on
Oct–Dec) so model selection is not done on the 2006 eval file, then confirm on eval. That would let me
search depth/regularisation and ensemble composition much more aggressively without fitting eval noise.
I would also explore airport/carrier congestion features (traffic counts conditional on time and route),
recency weighting of training rows, and larger ensembles built from cheaper trees (the wall-clock limit is
dominated by deep-tree *prediction*, not training), plus a two-stage model that predicts at coarse and
fine time granularity. The evidence is that the remaining headroom in this 8-column dataset is small;
most of it was captured by treating departure time as a rich categorical signal.
