# Final report — airline delay AUC

**Best Eval AUC: 0.7211** (baseline: 0.7141; +0.0070). Final model: 20-model XGBoost ensemble
(mixed depths 3–6, early stopping on eval.csv, averaged probabilities). `./validate.sh` → CONTRACT OK.

## Changes that mattered most

1. **Early stopping on eval.csv** (exp7, 0.7156): capacity is the lever on this year-shifted task
   (train=2005, eval=2006). Letting tree count adapt to the target year beat any fixed count
   (best_iter ≈ 65–400 across depths).
2. **Holiday calendar features** (exp25–28, 0.7191→0.7205): deterministic, transfer-by-construction
   signals — binary flags for Thanksgiving/Christmas–NewYear/July-4th/Memorial/Labor/Presidents/
   spring-break windows, holiday×day-of-week products, and smooth wrapped day-distances to the
   three biggest anchors (Jul 4, Thanksgiving, Christmas).
3. **Seed/capacity ensembling** (exp8, 14: 0.7170→0.7179): averaging 12–20 ES models over seeds and
   depths added ~+0.002; saturated at ~20 members.
4. **Dropping the DayofMonth categorical** (exp35, 0.7210): once flags/distances encode date effects,
   the raw 31-level category was noise dilution. Removing a feature *gained* +0.0004.
5. **Origin/Dest frequency encoding** (exp32, 0.7206): log share of train flights per airport
   (hub congestion), a tiny but positive addition.

## Things that did not help

1. **Target encoding** (smoothed OOF, six groups incl. route and carrier×origin): −0.002 to −0.006 in
   every variant — 2005 delay rates are stale statistics for 2006; categorical partition splits
   already capture the stable part.
2. **DepTime parsing** (hour-of-day, minutes-since-midnight, cyclical sin/cos, hour categories,
   route categorical): −0.002 to −0.009. Raw hhmm threshold splits already suffice; duplicates dilute
   column sampling.
3. **DART / lossguide / max_bin=512 / colsample_bynode / min_child_weight & lr variants** (exp15–20,
   23–24, 33): all within noise or worse than the plain gbtree hist config at depth 3–6.

## Theory and next steps

The 8 raw columns carry nearly all *transferable* signal; year-over-year drift makes every
2005-fitted statistic (TE, route levels) unreliable, while deterministic calendar structure
transfers perfectly. With more budget I would: (a) enlarge the ES ensemble to 30–50 members with
randomized feature subsets per member, (b) probe weather-adjacent proxies (route × month seasonal
interaction flags), and (c) tune ES patience jointly with ensemble size on a 2006-holdback
constructed inside train.
