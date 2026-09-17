# Final Report

**Best Eval AUC: 0.7565** (experiment #14, commit `2d31563`; equal to #13 `7182174`, which #14 supersedes
with a memory-safe streaming implementation of the same model).

Progression: 0.7141 (baseline) → 0.7425 → 0.7470 → 0.7498 → 0.7504 → 0.7512 → 0.7536 → 0.7538 → 0.7545 →
0.7561 → **0.7565** across 15 official experiments.

## Changes that mattered most

1. **Time-of-day categoricals + hour×carrier interaction** (exp #3, +0.03): HourCat (24 levels), TodCat
   (96 × 15-min buckets), and HourCarr (~480-level categorical) as identity categoricals. Departure-time
   effects are the dominant signal and bucketed categoricals capture them far better than raw hhmm ints.
2. **DepTime jitter augmentation** (exp #4-#5, +0.003): appending a ±10-minute jittered copy of the
   training data (jitter computed in minutes space, train-time only) acts as a strong regularizer against
   the 2005→2006 distribution shift.
3. **Congestion features** (exp #6-#7, +0.0015): log1p flight counts of Origin×hour, Dest×hour, and
   Origin/Dest×15-min-bucket computed on 2005. Airport-time load transfers to 2006.
4. **Route-typical departure time + deviation** (exp #8, +0.0025): per-route circular mean (sin/cos) and
   median departure minute, plus this flight's circular deviation from the route median. Flight schedules
   are structural year-to-year, so this transfers; also carrier-hour density and carrier-hub counts.
5. **Ensembling + slow learning + test-time augmentation** (exp #4/#9-#14, +0.007 cumulative): 3 diverse
   members (d8/lr0.03/cs0.6; d10/lam30; one-hot time buckets), lr 0.03 with ~800 rounds, colsample 0.6,
   prob-averaged; plus 9-view TTA (original + 8 deterministic ±5-min jitters) averaged at predict time.

## Things that did not help

1. **Early stopping / validation-based tuning on 2005**: useless because of the distribution shift
   (2005-val AUC 0.75-0.76 while 2006 eval was 0.70-0.72); fixed round counts just past the eval-curve
   peak transfer better.
2. **Target/route encodings and route identity categoricals**: overfit 2005 idiosyncrasies; also cyclical
   month/day-of-week features, day-of-week interactions, and 5-minute time buckets.
3. **Deeper search tricks**: MonthCat/DayofMonth as categoricals (much worse: trees need the ordinal
   encoding), slimming "redundant" time features, rank-averaging vs prob-averaging, a 4th ensemble member
   (saturates at 3), seasonal (origin×month) congestion, and TTA amplitude > 5.

## What I would try with more budget

The model is a 3-member XGBoost ensemble on 2× jitter-augmented 2005 data with schedule/congestion
features and 9-view TTA. With more budget I would: (a) build proper 2005 out-of-fold predictions and try
stacked generalization with a logistic blender, checking whether it beats simple prob-averaging across
years; (b) search for more *structural* 2005→2006-stable features, e.g. per-origin scheduled-departure
banks (multimodality of the origin departure-time distribution) and per-route schedule variance, since the
route-median feature family gave the single largest recent gain; (c) tune TTA jointly with training
(e.g., matching train-jitter and TTA amplitudes, or feature-noise views beyond DepTime); and (d) verify
gains against a 2005 holdout-year split to reduce the risk of tuning to the 100k eval slice.

**Status: budget management.** 25 experiment slots and ~146 minutes remained, but the CPU ledger
(~16.5k/18000 s) only allowed ~3 more official runs, and the last one (TTA ×13) timed out at the 120s
module cap and was reverted. Stopped experimenting per the finalization rules.
