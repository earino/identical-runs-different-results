# Final Report — autoresearch XGBoost (airline)

**Best Eval AUC: 0.7577** (experiment #40, commit `491e4c2`, contract-validated via `./validate.sh` → `CONTRACT OK`).
Baseline was 0.7141, so the loop added **+0.0436 AUC** over 40 experiments.

## Setup
`train.py` does all feature engineering inside `prepare(df)`, which `predict_proba(df)` also calls, so the
hidden holdout gets the identical pipeline. All encoders/statistics are fitted on `data/train.csv` only.
The final predictor is an average of 12 diverse `XGBClassifier` models.

## Changes that mattered most (most → least)
1. **Out-of-fold target (delay-rate) encoding at fine time-of-day resolution.** Encoding the mean target for
   keys like `route × 5-min bucket`, `carrier × route × 5-min bucket`, `origin × 15-min`, etc., using 5-fold
   OOF values for training and full-training maps at inference. Going hour → half-hour → quarter-hour → 5-min
   produced the largest single jumps (0.7358 → 0.7475 → 0.7530 → 0.7534). Time-of-day is by far the dominant
   signal, and schedule slots are stable across years.
2. **Delay-propagation features.** For each row, the encoded delay rate of the same `route`/`carrier-route`
   (and origin/dest/carrier) in the *previous* and *next* 5-minute bucket. Captures cascading delays; added
   +0.0017 and +0.0005 (0.7555 → 0.7572 → 0.7577).
3. **Congestion / input-count encodings.** Counts of flights per `(Origin, hour)`, `(Dest, hour)`,
   `(DayOfWeek, hour)`, `(Origin, Month)`, `(route, hour)`, plus hub size (`n_carriers_route`, `n_dest_origin`).
   Pure input statistics, so they transfer across years (0.7159 → 0.7285 across this line of work).
4. **Dropping the raw high-cardinality categoricals** (`Origin`, `Dest`, `UniqueCarrier`) after encoding them.
   They were memorizing 2005 and hurting 2006 generalization; removal alone gave +0.0024.
5. **Model regularization + a diverse ensemble.** `min_child_weight=50`, `reg_lambda=5`, depth 3–5 mix,
   colsample/subsample varying, `max_bin=512`, 800 trees at lr 0.04, averaged over 12 seeds. Ensembling and
   regularization each contributed small but consistent gains (0.7153 → 0.7168 → … → 0.7358).

## Things that did NOT help
- **Simply adding more trees to the baseline** (30 → 100 → 500 at lr 0.1): eval AUC *fell* (0.7141 → 0.7125 →
  0.7073) because the model overfit the 2005 slice and the year shift is severe.
- **TE interaction "residual" features** (`pair rate − marginal rate`): 0.7256, a large regression.
- **Very fine 2-minute buckets** (0.7523 vs 0.7555) and **extra day-of-week / route-month interactions**
  (0.7353, 0.7267, 0.7553) — sparse keys just add noise. Depth 6 also overfit (0.7249).

## What I would try with more budget
- **Hierarchical / neighbor-smoothed target encoding**: shrink each fine bucket toward its coarser parent
  (5-min → quarter → hour) instead of the global prior, and/or smooth each bucket with its temporal neighbors,
  so fine resolution is kept without the sparsity penalty that made 2-minute buckets fail.
- **Recency/domain adaptation**: since eval and holdout are 2006 while train is 2005, weighting training rows
  and TE fits toward late-2005 months, or fitting an explicit year-shift correction (e.g. per-airport rate
  deltas), could recover the residual distribution shift.
- **More propagation structure**: model the sequence of departures from an airport (a "delay clock" feature),
  and encode the previous flight's delay potential rather than only the rate.
- **Stacking** the 12 XGBoost members with a small out-of-fold meta-model (still XGBoost) instead of a plain
  probability average, and a proper nested temporal validation scheme to tune the number of trees without
  touching `eval.csv`.
