# Final Report — airline delay AUC

Best Eval AUC: **0.7279** (HEAD = exp36/exp40 commit 22cafeb, `CONTRACT OK` via validate.sh).

## Final model
50-member XGBoost seed ensemble; each member: 300 trees, lr 0.05, depth 8,
min_child_weight 20, gamma 2.0, subsample 0.5, colsample_bytree 0.5, hist,
native categoricals for UniqueCarrier/Origin/Dest. Features: Month, DayofMonth,
DayOfWeek, DepTime, Hour, Minute, Distance, log1p(Distance) as numerics; raw
c-strings for the three carrier/location categoricals. All engineering inside
`prepare()`; category levels fitted on train only.

## Changes that mattered most
1. **Seed ensembling** (exp11-12): averaging 10-30 identical-config members with
   different seeds: 0.7138 -> 0.7190. The single biggest robust gain.
2. **Dropping Month/DayofMonth/DayOfWeek categoricals** (exp16): 0.7190 -> 0.7242.
   2005 seasonal-amplitude memorization actively hurt 2006 AUC (delay-rate
   seasonality flips between years, e.g. April 0.41 -> 0.48).
3. **Heavy bagging, stronger regularization, slower learning** (exp21/22/24/31):
   depth 8 + mcw 20 + gamma 2 + lr 0.05x300t + subsample/colsample 0.5:
   0.7242 -> 0.7279. More smoothing = better cross-year transfer.
4. **Minimal numeric feature set** (exp27): dropping the four sin/cos cyclical
   features simplified and improved: +0.0009.
5. Calendar/datetime columns kept only as *plain numerics* (the c-<n> decode +
   hour/minute split); native categorical handling for airports/carriers.

## What did not help
- Target encoding of Origin/Dest/Carrier (OOF, smoothed): 0.7123 — 2005 rate
  levels do not transfer to 2006 (per-level rates correlate only ~0.85).
- Route/Origin-x-Hour/Month-x-Hour/DepTime interaction categoricals: equal or
  worse than the simple feature set (Route as native category also crashes on
  unseen levels).
- More/longer/deeper single models: ES-tuned depth-8 2400-tree model 0.7095;
  60 seeds = 30 seeds; lr 0.05x300t with s/c 0.7 0.7188 vs 0.7190.
- Random-split early stopping as model selection: valid 0.7569 but eval 0.7095
  (temporal shift); random CV overstates transfer.

## With more budget
- Calibrate/blend per-month (or per-quarter) rank-normalized predictions to
  neutralize seasonal amplitude drift while keeping within-month ordering.
- Bigger bagged ensemble of shallower (depth 6-7) strongly-regularized members
  at lr 0.03; bagged 0.5/0.5 was still improving at the margin.
- Learned monotonic constraints on Hour/DepTime; quantile-bucketed DepTime
  features; per-carrier hour interactions as numerics (not categoricals).
