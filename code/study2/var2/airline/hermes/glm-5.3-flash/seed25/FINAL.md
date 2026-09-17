# Final Report — airline delay (XGBoost, hidden-holdout AUC task)

**Best Eval AUC: 0.7321** (baseline 0.7141, +0.0180). Final commit `cfdfebe`, validated
(`CONTRACT OK`, predict_proba reproduces 0.7321 with the target column dropped).

## Final model
Single `XGBClassifier`, depth 4, 2000 trees, lr 0.03, hist, categorical splits enabled,
trained on all 100k rows of 2005. Feature set: raw categoricals (Month, DayofMonth,
DayOfWeek, UniqueCarrier, Origin, Dest), Distance + log1p(Distance), time-of-day as both
numeric (DepHour, DepMin, sin/cos) and categorical buckets (DepHourCat 24, DepHalfHour 48,
DepQuarterHour 96), day-of-month sin/cos, minute-in-hour, on-slot flag, and two categorical
interactions (Carrier×Hour 480 levels, Carrier×HalfHour 960 levels).

## Changes that mattered most
1. **Time-of-day re-encoding** (+0.003 from #7→#11, then more): raw hhmm DepTime is nearly
   useless to shallow trees (#8: 0.5956 at depth 3); extracting hour/minute and adding
   cyclical sin/cos was the single biggest feature win.
2. **Categorical hour buckets instead of numeric hour** (+0.0010, #19): per-hour delay
   levels are non-monotonic, so arbitrary per-level effects beat range splits; then depth 4
   became optimal (+0.0015, #20).
3. **Carrier×Hour interaction** (+0.0082, #27): the largest single jump. Carrier bank
   schedules vs. time of day transfer across years, unlike anything month- or
   geography-keyed. Finer Carrier×HalfHour (+0.0005, #32) and Dow×Hour (+0.0008, #31)
   added on top.
4. **Shallow-and-long instead of deep-and-short**: depth 3→4 with 2000 trees at lr 0.03
   (#22, +0.0010) beats every deep configuration tried; the 2005→2006 shift punishes
   2005-specific capacity (depth 8: −0.003).
5. **DepQuarterHour** (+0.0006, #40): the finest time bucket still helped; schedule
   granularity is the dominant signal in this dataset.

## Things that did NOT help
1. **Target encoding of carrier/origin/dest** (#12): exactly equal AUC — 2005 delay *rates*
   per airport/carrier don't transfer to 2006.
2. **Subsample/colsample bagging, any form**: single-model subsample (#5), 5-model bag
   (#15) both clearly worse; 2-seed averaging without subsampling is exactly neutral (#39)
   because hist trees are deterministic at fixed data.
3. **Route identity / route frequency / Origin×Hour** (#24, #25, #28): Origin×Dest pairs
   and 282-airport×hour cells memorize 2005 traffic patterns; 0.6988 at worst.
4. **Month×Hour** (#30, 0.7246) and rare-airport regrouping (#37, 0.7299): monthly
   seasonality and cardinality reduction both fail the year shift.
5. **Regularization knobs** (min_child_weight=50 #17, gamma=1 #35, max_bin=1024 #26):
   neutral to harmful; the capacity limit is set by the distribution shift, not by tree
   regularization.

## With more budget
Grid the interaction cell more systematically: Carrier×QuarterHour, Carrier×DayOfWeek×Hour
(small level counts each, maybe sum-composed rather than cross-product), and
Dest-side features (arrival congestion at Dest may matter even for departure delay via
turnaround pressure). I would test time-interactions keyed on *slot position within the
day at each carrier* rather than absolute hour (robust to schedule changes year to year),
try quantile-binned DepMin categories, and revisit ensembling only with heterogeneous
members (different depths/feature subsets rather than seeds). A proper time-based internal
split (e.g. train on months 1–9, validate on 10–12) would make model selection cheaper and
less noisy than repeated eval.csv probes.

## Selection-noise caveat
eval.csv is 100k rows; AUC differences < 0.0005 are within selection noise. Gains kept were
all ≥ 0.0002 and mechanistically explicable; the final +0.0006 (#40) is the weakest-kept
change but sits on a consistently improving axis (finer time buckets).
