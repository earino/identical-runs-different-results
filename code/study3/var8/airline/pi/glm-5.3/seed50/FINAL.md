# FINAL — airline delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7612** (experiments #9 and #14; HEAD is #14: 5-model bag, d24, mcw0, max_bin=64,
cs=0.25, 150 trees @ lr 0.12 — same AUC as the 4-model version but one more bagging member, faster
(91s vs 102s) and therefore a safer timeout margin for the hidden-holdout scorer).

Baseline (30 trees, raw columns) was 0.7141 → **+0.047** over the session.

## What mattered most (in order of impact)

1. **Understanding the time-shift**: train=2005, eval/holdout=2006. Eval AUC peaks at a small
   "total boost" and *declines* with more capacity — random-split early stopping chases 2005 noise
   (it happily picked 596 trees while the 2006 peak was ~120). Fix: fixed small round count, no
   random-split early stopping.
2. **Random-forest-style XGBoost**: `colsample_bytree=0.25`, `min_child_weight=0` (fully-grown trees),
   `max_depth=24`, `max_bin=64`, and a 4–5-seed bagged ensemble. Decorrelation was worth far more
   than any single hyperparameter tweak (0.727 → 0.753 over several steps).
3. **Volume counts**: log flight-frequency in the training sample for route / origin / dest / carrier
   (busier operations delay more): +0.003.
4. **Schedule-deviation features** (the biggest feature win, +0.006 total): deviation of a flight's
   departure time from its *route's* / *origin's* / *carrier's* typical time (circular difference),
   and of its estimated arrival time (DepTime + distance/500mph + 1h) from the *destination's* /
   *route's* / *carrier's* typical arrival time — flights off their route's normal schedule behave
   differently; evening arrival banks propagate delays.
5. **DepTime decomposition + cyclical encoding**: hour, minute, minutes-of-day, sin/cos of tod.

## What did not help

- **Route as a categorical (4.2k levels) or route target-encoding**: −0.005; hurts badly.
- **Random-split early stopping** (overfits 2005; picks too many rounds): −0.015 vs the tuned point.
- **Numeric duplicates of the calendar categoricals** (month_n/day_n/dow_n): −0.003.
- **Extra volume counts** (origin×month, origin×dow, route×dow, hour) and unseen-value indicators: dilute
  the feature pool under heavy colsampling → −0.001 to −0.003.
- **Row-bagging per seed** on top of per-tree subsample: −0.0016 (just loses data).
- **colsample retune upward** (0.3–0.5) as features grew: 0.25 stayed optimal.

## With more budget I would try

A **DART / more aggressive decorrelation sweep** at the RF-style config (rate_drop ~0.1), month-based
(temporal) validation inside 2005 to replace fixed round counts, **interaction-aware binning of
tod_dev_route × hour**, replacing the native categorical splits with **smoothed hierarchical
airport-cluster targets** (region/busyness tiers instead of 300 raw airports), and a proper search
over `max_bin`/`max_leaves` with a lossguide policy. I would also try squeezing a 6th bag member by
reducing trees further (the last seeds still add a little: the 5-seed run matched eval AUC with 25%
fewer trees and more timeout safety).
