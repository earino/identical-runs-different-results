# FINAL — airline dep-delay XGBoost benchmark

**Best Eval AUC: 0.7411** (experiment #39, commit b69673a) — up from the 0.7141 baseline.

## The 5 changes that mattered most

1. **Numeric/cyclical feature extraction** (exp 6): `c-<n>` strings -> ints, sin/cos for month/day/dow,
   hour-of-day from DepTime, log-distance. Kept Origin/Dest/Carrier as native categoricals.
   0.7141 -> 0.7181. Crucially this included *dropping* the naive Origin+Dest route string (see below).
2. **Decorrelated bagged ensemble** (exp 13-19): many XGB members (final: 42) averaging probabilities,
   with per-member diversity over seeds, depth grid 6-9, subsample grid 0.60-0.85, colsample 0.7-0.9,
   min_child_weight 1/3/5, lambda 1/5. Capacity per member was the key limiter: ~220 trees @ lr 0.07
   is the sweet spot; 300+ trees overfit the 2005 slice and hurt 2006 transfer. 0.7181 -> 0.7246.
3. **Label-free count/structure features** (exp 21, 24, 25): log flight-volume per Origin/Dest/Route,
   per Origin/Dest/Carrier x Hour, per Origin/Dest x DayOfWeek, Carrier x Origin/Dest. 0.7246 -> 0.7276.
4. **Fine-grained congestion counts** (exp 31-33): log flight-volume per airport x 30-min and then
   10-minute time bucket (mod 1440, so 24:00-26:00 wraps correctly). 15-min hurt slightly at 5-min
   granularity. This was the single biggest feature win: 0.7276 -> 0.7365.
5. **Feature-view (policy) diversity in the ensemble** (exp 37-39): each member trains on a different
   column subset - all features / no categoricals / no plain counts / no congestion buckets /
   time+congestion only / no cyclical features. Averaging structurally different views beat any
   amount of seed jitter: 0.7365 -> 0.7411, and it trains *faster* (95s).

## The 3 things that did not help

1. **Origin+Dest route string as a categorical** (exp 4, 20): -0.009 as a single model, -0.015 inside the
   ensemble. 4198 levels over 100k rows memorizes 2005 schedules that do not transfer to 2006.
2. **Target encodings of any kind** (exp 10, 11, 29): marginal TEs (Origin/Dest/Carrier/Route/Hour) and
   interaction TEs (Origin x Hour, Dest x Hour) all hurt, from -0.0006 to -0.007. 2005-calibrated label
   statistics shift in 2006 and their split thresholds then misrank rows. Label-free counts + the trees'
   own categorical splits do it better.
3. **More boosting rounds / early stopping on an internal split** (exp 2, 3, 17, 28, 34): a random 2005
   validation split stops boosting at ~42 trees (0.7067) because it rewards 2005-specific fit;
   >300 trees per member or capacity re-tuning after the count features all landed within noise or worse.
   Seasonal Origin/Dest x Month counts also hurt (-0.001, monthly schedules differ across years).

## What I would try with more budget

The two strongest levers turned out to be (a) schedule/congestion structure that is stable across years and
(b) averaging over structurally different views rather than tuning a single model. Next I would attack the
sparsity wall of the congestion features: with 1M holdout rows but only 100k training rows, the 10-minute
airport buckets are already ~7 rows/cell, so finer time resolution stops paying; I would instead try
smoothing counts across adjacent buckets (e.g. a 3-bucket moving sum, which keeps hub density but shrinks
cell noise), dow-resolved congestion at 30-minute granularity, and fitting the volume maps on *train+eval
features only* if the rules allowed (labels would stay untouched). On the ensemble side I would expand the
policy space (policies that drop Distance entirely, weekday-only or weekend-only members, members trained
on month-blocked row subsets), rank-average instead of probability-average, and calibrate the mix of
policies by out-of-fold weight fitting. Finally, an honest internal time-forward validation (train Jan-Sep
2005, validate Oct-Dec 2005) would give a transfer-honest early-stopping signal that the random split
could not, which would let me safely push member capacity again.
