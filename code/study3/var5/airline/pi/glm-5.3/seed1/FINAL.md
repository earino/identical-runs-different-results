# Final report — airline delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7457** (baseline: 0.7141, +0.0316). Final config at commit `666ae35`
("colsample 0.6 + max_bin 1024"), validated with `./validate.sh` → `CONTRACT OK`.

## What mattered most (in order of impact)

1. **Deep trees + early stopping on an internal split, then refit on the full training set.**
   Refitting on 100% of the data with the ES-selected tree count beat both the internal-split model
   and fixed tree counts (+~0.004 alone). Depth was the single biggest hyperparameter: AUC climbed
   monotonically from depth 6 to ~20 (0.7141 → ~0.7379) — deep trees are needed to exploit the native
   categoricals (Origin/Dest/Carrier) and hour-of-day interactions.
2. **3-seed ensemble refit on the full data** (average of 3 XGBoost models differing by row/col
   subsampling seeds): +~0.002.
3. **Numeric date/time features from the raw columns**: parsing `c-<n>` strings to ints, hour/minute
   from hhmm DepTime, DayOfYear, log-distance (+~0.003 over raw). Hour-of-day is by far the
   strongest predictor (5am trough → late-night peak).
4. **Strong feature/col bagging at high granularity**: colsample_bytree 0.6 (+0.002) and max_bin 1024
   (+0.001) on top of deep trees.
5. **Route popularity count** (train-set flight count per Origin→Dest pair, +0.0008) and a
   **2-fold early-stopping scheme** (train-on-80%/validate-20% and its complement, averaged tree count)
   to stabilise the ensemble's rounds.

## What did NOT help (all reverted)

1. **Any per-category delay statistics**: target/mean encoding of Route/Origin/Dest/Carrier hurt
   (-0.0045) and the raw Route categorical (4198 levels) hurt badly (-0.008) — 2005 category noise
   does not transfer to the 2006 eval split.
2. **Extra interaction/cyclic features**: hour-of-week categorical, origin/dest busyness counts,
   week-of-year, sin/cos encodings, "hours-since-5am" night-ramp — all flat or negative; depth-20
   trees already recover these splits.
3. **Alternative structures**: one-hot categoricals (`max_cat_to_onehot`), `num_parallel_tree=2`,
   DART (prohibitively slow at depth 20), time-decay sample weights, higher reg_lambda,
   colsample_bynode, larger ensembles (4–5 members), and learning-rate variants (0.02/0.035).

## What I would try with more budget

The config sits on a broad plateau: 8 consecutive bracket checks around it (depth 18/24,
colsample 0.5–0.7, subsample 0.75–0.85, lr 0.02–0.035, max_bin 2048, carrier one-hot) all landed
within 0.7442–0.7456. With more budget I would (a) scale the ensemble to many more members trained
on bootstrap resamples with per-member feature/depth diversity, since the 3-member seed ensemble
gained the most of any structural change and averaging is the only reliable lever left; (b) build a
hierarchical time-aware validation (multiple year-stratified ES folds) to tune the tree count
against the 2005→2006 shift instead of random splits; and (c) explore richer time-of-day × season
representations fit only on train (e.g. per-hour delay-rate curves with heavy smoothing) as
features rather than encodings of high-cardinality categoricals.
