# Final report — airline delay AUC

**Best Eval AUC: 0.7378** (experiment #40, commit `47a4b27`), up from the 0.7141 baseline.
All 40 experiments used; `./validate.sh` prints `CONTRACT OK`.

## Changes that mattered most

1. **One-hot encoding of all categoricals** (`max_cat_to_onehot=500`, exp #17, +0.0047). XGBoost's
   default partition-based categorical splits overfit the time-separated 2005→2006 shift badly; giving each
   airport/carrier/day/month its own indicator transfers much more robustly.
2. **Deep ensemble of XGBoost models** built up over exp #12–#39. Depth range grew 4→12, with per-model
   early-stopping splits and 8 distinct seeds. Each depth increase gave a small, consistent gain; the final
   config is depths {9,10,11,12} × 8 seeds, probabilities averaged. Contributed ~0.008 cumulatively.
3. **Carrier × departure-hour interaction** (exp #40, +0.0030). One-hot `UniqueCarrier_hour` was the single
   largest late gain, mirroring the value of explicit one-hot categorical structure.
4. **Time-of-day features** (`dep_hour`, `dep_minute`, `dep_since_midnight`, `dep_hour` as a one-hot
   categorical, 15-minute buckets) and **day-of-year** (exp #5, #14, #27, #28). Departure time is the
   dominant delay driver; hour-as-categorical beat hour-as-numeric (+0.0016).
5. **Looser leaf regularization**: `min_child_weight` 5→1 and `reg_lambda` 1.0→0.3 (exp #32, #33, #35).
   Rare one-hot categories need fine splits, so relaxing these helped once one-hot was in place.

## Things that did not help (reverted)

- **Route (Origin→Dest) as a feature**: partition form −0.015, fully one-hot form −0.001. Route delay
  propensity does not transfer across the year boundary; Origin and Dest one-hots are enough.
- **Smoothed target encoding** of carrier/origin/dest/route: −0.011. It leaks the training label and the
  model over-relies on it.
- **Frequency counts, cyclical sin/cos, holiday flags, distance bins, week-of-year one-hot**: all neutral
  to negative. Seasonality is already captured by month one-hot + numeric day-of-year.
- **Deeper-than-needed regularization** (`min_child_weight=20`: −0.005) and stronger feature subsampling
  (`colsample=1.0`: −0.001).

## What I would try with more budget

A next step would be out-of-fold target encoding of `Origin`, `Dest`, `UniqueCarrier` and `carrier×hour`
(so the training rows never see their own label), which should add durable airport/carrier delay-propensity
signal without the leakage that sank plain target encoding. I would also test `grow_policy="lossguide"`
with a large leaf budget as an alternative to fixed depth, and, since per-model split seeds helped, bagging
over more resampled training subsets. Finally, a small amount of seasonal-sample weighting (weighting late
2005 months more, as the evaluation period is early 2006) might improve temporal transfer.
