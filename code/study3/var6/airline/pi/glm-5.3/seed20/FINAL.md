# FINAL — airline dep-delay XGBoost

**Best kept Eval AUC: 0.7595** (12-member bagged deep-tree ensemble, 73s runtime). A 20-member
variant measured 0.7597 but ran at exactly the 120s wall cap; it was abandoned in favor of the
statistically indistinguishable 12-member config with a 47s safety margin (the scorer re-runs
train.py, and a timeout there would score 0).

Baseline (30 shallow trees, raw strings) was 0.7141.

## Changes that mattered most (in order of impact)

1. **Deep-tree bagged GBM ensemble** (12 × XGBoost, max_depth 60, subsample 0.8, colsample_bytree 0.5,
   few boosting rounds): deep trees capture hour×carrier×airport interactions while row/column
   bagging + seed averaging prevents each tree from memorizing 2005. This took AUC 0.7141 → 0.7521.
   Everything "big" (deep) with heavy bagging beat every conventional-GBM setting I tried.
2. **Engineered time features**: parse `c-N` strings to ints, DepTime → hour/minute/tod + sin/cos
   encoding, log-distance. Time of day is the dominant delay factor (5am ≈ 4% delayed, 1am ≈ 85%).
3. **Structural volume features** (train-only counts): origin/dest/route/carrier traffic volume and
   origin×hour, dow×hour, carrier×hour congestion. Unlike target encodings, volumes are schedule
   attributes that repeat year-over-year, so they transfer 2005 → 2006: +0.005 AUC in two steps.
4. **Recency weighting** of training rows (linear ramp, Dec-2005 weighted 3× vs Jan-2005): the
   2005→2006 shift is a real distribution drift, and later months are more representative of 2006.
5. **Fewer boosting rounds** (30) — under heavy bagging and deep trees, extra rounds only memorize
   2005; the optimum rounds dropped every time model capacity or features grew.

## What did NOT help (all reverted)

1. **Target (delay-rate) encodings** of carrier/origin/dest/route/hour — they inject 2005-specific
   delay rates and hurt transfer (0.7124 vs 0.7198).
2. **Route (Origin_Dest) as a categorical** — 3000+ categories = 2005 route memorization; dropped
   AUC by ~0.01. Same for hour×dow and hour×carrier interaction categoricals.
3. **Early stopping on a 2005 validation split** — internal val AUC climbs to 0.77 while 2006 AUC
   falls from 30 rounds on; picking rounds by random-split early stopping is actively harmful here.
   (Month-block validation inside 2005 also failed to mimic the year shift.)
4. Also-rans: lossguide trees (0.7423), per-node colsample, mixed-depth/mixed-policy ensembles
   (diversity from weaker members always diluted the average), max_bin=512 (timeout), mCW≥2,
   cs=0.4, ss=0.7, logit-space averaging (identical), second batch of sparse volume features
   (origin×month etc.), temporal month-dropout bagging, more members (n14/n16/n20 ≈ noise).

## With more budget I would

focus on the two remaining structural unknowns rather than more hyperparameter noise: (a) the
2005→2006 drift itself — e.g. fit a "drift-robust" blend by training one ensemble per training-month
block and weighting members by their agreement with the stable structural features, or use a
small, strongly-smoothed calibration on eval.csv to re-weight members; (b) richer *non-target*
schedule features that repeat across years (route×hour frequencies, hub connectivity stats,
day-of-week × origin congestion with heavier smoothing), since the volume family was the single
largest late-stage gain. I would also like a proper uncertainty estimate on eval-vs-holdout
differences — several late decisions were made on ±0.0003 noise, and 5-fold by-month bagged
retraining would give far more reliable keep/drop decisions than the single 100k-row eval slice.
