# FINAL — airline dep_delayed_15min (autoresearch XGBoost)

**Best Eval AUC: 0.7387** (commit `1b98308`, experiment #31) — up from 0.7141 baseline (+0.0246).
Contract validated: `CONTRACT OK` (predict_proba reproduces 0.7387 on raw frames).

## Final train.py shape
- Features (all inside `prepare()`, fit on train only): raw columns (Month/DayofMonth/DayOfWeek/DepTime/
  UniqueCarrier/Origin/Dest/Distance as native categoricals + numeric) plus:
  - `hour` (24-level cat), `qhour` (96-level 15-minute-bucket cat)
  - `tod_sin`/`tod_cos` (cyclic minutes-since-midnight)
  - `carrier_hour` (UniqueCarrier x hour interaction cat, 480 levels)
- Model: single `XGBClassifier(n_estimators=2500, max_depth=6, learning_rate=0.03, max_bin=512,
  min_child_weight=5, tree_method=hist, enable_categorical=True)`, ~20s.

## Changes that mattered most
1. **Capacity: 300 -> 2500 trees at lr 0.03** (exp 6/7/26): +0.005. Fixed-tree-count training at low lr;
   early stopping on random/month splits always undershot badly.
2. **`max_bin=512` + `min_child_weight=5`** (exp 17): +0.0015. Finer hist cuts on DepTime/Distance.
3. **`hour` as its own 24-level categorical** (exp 3): +0.0015 over relying on raw DepTime splits.
4. **`carrier_hour` interaction categorical** (exp 23): +0.0098 — the single biggest jump. Carrier-specific
   schedule banks (dawn regional feeds vs evening hubs) have distinct delay profiles.
5. **`qhour` 15-minute buckets** (exp 27): +0.0059. Sub-hour delay structure is real; 96 levels stay dense
   enough to generalize across the 2005->2006 shift.
6. **`tod_sin`/`tod_cos` smooth cyclic encoding** (exp 31): +0.0011 on top of the buckets.

## What did not help (all reverted)
1. **Target encodings**, both naive in-sample (0.7105) and correct cross-fit OOF (0.7137): 2005 delay rates
   per route/airport do not transfer to 2006; native categorical splits were strictly better.
2. **Every interaction with more than ~500 levels**: route (4.2k) 0.6958, origin/dest x hour 0.7149,
   carrier x qhour (1.9k) 0.7285, origin x carrier 0.7316 — sparse cells memorize 2005.
   Only dense interactions (carrier x hour) survive the year shift.
3. **Sampling/ensembles/structural**: subsample 0.7-0.8, colsample 0.7, lossguide, seed ensembles
   (hist is near-deterministic, so same-config seeds add nothing), and time-aware early stopping
   (month-split validation kills seasonality, stops at 155 trees, 0.7079).

## What I would try with more budget
The winning pattern is "dense, physically-motivated time features at multiple resolutions." Next I would
(1) sweep bucket resolutions systematically (5/6/12/20-minute) and their interactions with DayOfWeek,
since carrier schedule banks are keyed to the weekly cycle; (2) build *out-of-time* validation from the
last 3 months of 2005 to tune tree count honestly instead of fixed 2500; (3) replace the single model with
a 3-seed x {qhour, hour, both} feature-subset ensemble, which creates genuine diversity where seed-only
ensembles could not; (4) explore Origin-hub features (flights per airport per hour, airline market share
per airport) with strong smoothing, since volume signals were neutral-to-positive but under-explored;
(5) calibrate `min_child_weight`/`gamma` jointly against the time-shifted validation, as regularization
was only probed at two points.
