# Final Report — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7247** (baseline 0.7141, +0.0106) — commit `16626c1`.

## Final model

10-model XGBoost ensemble (probability-averaged), members: depth 4-6, lr 0.05,
min_child_weight 20, gamma 1, reg_lambda 10, subsample/colsample 0.7-0.9,
2000 trees with early stopping (patience 100) on data/eval.csv.

Features: Month/DayofMonth/DayOfWeek parsed to numeric, DepTime split into
hour (with `hour = (DepTime//100) % 24` wrap-around fix for times > 2359) +
minute + cyclical sin/cos of time-of-day, log1p(Distance); categoricals kept
only for UniqueCarrier/Origin/Dest (native categorical splits).

## Changes that mattered most

1. **Dropping date categoricals for numeric+cyclical encodings** (+0.0037 then +0.0006):
   the c-N categorical splits on Month/DayofMonth/DayOfWeek memorized 2005-specific
   date patterns and did not transfer to 2006; numeric + sin/cos did.
2. **DepTime refinement** (+0.0019): wrap-around fix for DepTime > 2359
   (e.g. 2620 -> 02:20 next day, previously clipped to 23:59) plus a dep_minute feature.
3. **Shallow, regularized members** (+0.0016 vs baseline config): depth 4,
   min_child_weight 20, subsample/colsample 0.8 — variance reduction transfers
   better across the 2005->2006 year shift than capacity.
4. **10-model ensemble** (+0.0008 over single best): seeds x depths x sampling,
   probability-averaged; 20 models gave no further gain.
5. **Early stopping on eval.csv + lr 0.05** (+0.0004): members stop at 81-235 trees;
   eval (2006 slice1) is the best available proxy for the hidden 2006 holdout.

## Things that did NOT help

1. **Target encoding** (smoothed, m=30-200 for Origin/Dest/carrier/route/hour pairs):
   -0.0065 at m=30-60; even m=200 was only noise-level. 2005 conditional delay
   rates do not transfer to 2006.
2. **Cross categoricals** (route ~5k levels, org_dow, carrier x dow/month):
   -0.011 with high-card crosses, -0.002 even for low-card car_dow retested on
   the cleaned feature set. Interaction capacity overfits the train year.
3. **Frequency encodings, tail averaging, lossguide/mini-forest members,
   max_bin diversity, day-of-year feature**: all neutral-to-negative.

## With more budget

I would (a) tune the ensemble mix further (per-member objective-side diversity,
e.g. dart with small tree counts guarded against the 120s cap), (b) explore
smoothed shift-robust statistics computed *within* each year slice (e.g. rank /
quantile normalizations of airport/carrier delay propensities instead of raw
rates), and (c) revisit interactions as explicit numeric products (hour x
distance) rather than categorical crosses.
