# FINAL — airline delay (2005 train -> 2006 eval / hidden holdout)

**Best Eval AUC: 0.7517** (commit `ff4633c`, 8-seed lossguide XGBoost ensemble). Baseline: 0.7141.

## What mattered most

1. **Route x scheduled-hour target encoding (out-of-fold).** The single largest jump
   (0.7141 -> 0.7325). Smoothed (k=20) mean delay rate per `Origin|Dest|hour` key, computed on
   training labels only, with 5-fold out-of-fold values for the training rows and full-train
   statistics for prediction. Exact hour matters (6-hour buckets, route/route-month encodings were
   far weaker).
2. **Minute-of-hour features.** `dep_minute`, plus sine/cosine of the minute within the hour, and
   the numeric hour. Worth ~+0.003; schedule structure at minute granularity carries real signal.
3. **Label-free frequency and diversity encodings.** log counts of Origin, Dest, Route and Carrier,
   plus the number of distinct Dest per Origin, carriers per Origin and Dest per Carrier
   (~+0.003). These capture airport/route popularity and transfer across years.
4. **Dropping `Month`/`DayofMonth`.** Their delay base rates flip between the 2005 train and 2006
   eval slices (January 0.54 -> 0.44, April 0.41 -> 0.48), so the model generalises better without
   them. Origin/Dest/Carrier/DayOfWeek categoricals are kept.
5. **`grow_policy="lossguide"` (256 leaves) instead of depth-wise trees**, and averaging many
   seeds. Lossguide was worth ~+0.007 over depth 5-6; seed averaging added a further ~+0.001
   (5 mixed models 0.7503 -> 8 pure seeds 0.7517), and is also the change most likely to help the
   1M-row hidden holdout because it reduces variance.

## What did not help

- **Route as a native high-cardinality categorical** (0.7119, below baseline): it overfits 2005.
- **Extreme column subsampling without lossguide** (`colsample_bytree` down to 0.1-0.2 with depth
  trees) — it required thousands of weak trees and still saturated around 0.733.
- **Extra calendar/seasonal features** (`month`/`dayofmonth` cyclicals, weekend flag) and extra
  interaction encodings (route-month, route-dow, carrier-hour, count-of-route-hour) were all
  within +/-0.001 of the current feature set and did not transfer.

## With more budget

The model is feature-limited, not capacity-limited: eval AUC rises monotonically with lossguide
leaves up to ~256 and then plateaus, while target encodings of *scheduled* information dominate.
The most promising untried directions are (a) hierarchical/shrinkage target encoding that borrows
strength from route, origin and dest rates instead of one flat smoothed rate, (b)
propagation proxies built from scheduled timetables (e.g. how soon after another departure on the
same route/carrier a flight is scheduled), and (c) a proper XGBoost stack (out-of-fold predictions
of several diverse XGBoost models as meta-features) rather than plain probability averaging. Given
the strong 2005->2006 distribution shift, any future work should also cross-validate by year-slice
and prefer features that are stable across the per-month base-rate changes rather than tuning on
the single 100k eval slice.
