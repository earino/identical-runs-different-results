# Final report — airline departure-delay AUC

**Best Eval AUC: 0.7592** (commit `3a9719b`, 3-model depth-diverse XGBoost ensemble, weighted toward
the deeper models).
Baseline was **0.7141**. The model is trained on `data/train.csv` (2005) and evaluated on
`data/eval.csv` (2006); all feature engineering lives inside `prepare()` and was verified by
`./validate.sh` (`CONTRACT OK`, AUC 0.7592 with the target column removed).

## Changes that mattered most

1. **Time-of-day engineering.** `DepTime` (hhmm, including values >2400 for after-midnight
   departures) is the dominant signal. Parsing it into `DepHour`, `DepMinute`, `TOD` and a
   **30-minute categorical `TODbinC`** took a single model from ~0.58 (without time) to ~0.75.
   Treating the time profile as a categorical (arbitrary, non-monotone shape) beat numeric
   `TOD`/one-hot and every other bin size tested.
2. **High-capacity, strongly regularized trees.** `max_depth=16`, `learning_rate=0.03`,
   `colsample_bytree=0.5`, `subsample=0.8`, `reg_alpha=1.0`. Deeper trees were consistently
   better (14 < 16 < 18), and column subsampling + L1 gave a large, reproducible gain.
3. **Interaction categoricals with time.** `CarrTOD` (carrier x 30-min bin) was by far the
   most important feature; `OrigHr` (origin airport x hour) and `odc_freq`/`carrier_freq`
   (target-independent popularity counts) added smaller but consistent gains.
4. **Scheduled-arrival proxy.** Estimating duration (`Distance/7 + 30` minutes) and adding
   `ArrTODbinC`/`ArrHour` gave ~+0.002, capturing the fact that flights scheduled to arrive late
   in the day are more delay-prone.
5. **Small depth-diverse ensemble.** Training three XGBoost models (depths 14/16/18) with
   different seeds and averaging their probabilities added ~+0.002 over the best single model.
   Weighting the average toward the stronger deeper models (0.25/0.33/0.42) gave the final
   +0.0001 and the best recorded score.

## Things that did not help (and were reverted)

- **Route (`Origin_Dest`) categorical** and generic frequency encodings: both clearly hurt
  (route ~-0.017, frequency ~-0.015 in the early feature set), and route remained harmful later.
- **Target encodings** (full-train and 5-fold OOF, for carrier/origin/dest/carrier x hour):
  consistently worse than native categorical handling under the 2005->2006 time shift.
- **One-hot encoding, month/day-of-month/day-of-week as categoricals, leaf-wise
  (`lossguide`) growth, extra interactions** (dest x hour, weekday x hour, route x hour,
  carrier x month, distance x TOD, airport-size counts) and larger ensembles (4-5 models,
  12/14/16/18/20 depths): no gain beyond noise; the smaller 3-model ensemble was kept.

## What I would try with more budget

The score is near the ceiling reachable from these 8 columns: without tail numbers, actual
departure/arrival times, or upstream delay propagation, the only real signal is carrier/airport
identity, distance, and (strongly) time of day. The most promising untried direction is
**sequence/rotation features** — reconstructing aircraft rotations (or at least "delay of the
previous flight from the same origin/carrier/time bucket") from an ordered schedule, which is the
main physical driver of departure delay. Within the current setup I would also try a
time-aware/online target-encoding scheme (expanding-window per airport-carrier to respect the
2005->2006 shift), a proper out-of-fold **stacked XGBoost meta-learner** over several diverse
base models, and modest hyperparameter re-tuning around the ensemble now that the feature set
has changed (the earlier tuning was done before the interaction and arrival features existed).
