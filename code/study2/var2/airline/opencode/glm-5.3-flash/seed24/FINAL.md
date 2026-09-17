# Final Report — airline delay AUC

**Best Eval AUC: 0.7432** (baseline 0.7141, +0.0291 over 40 experiments; all budget used)

Final model: average of a 12-member XGBoost bag (hist, `enable_categorical`), base config
350 trees / depth 5 / lr 0.1 with per-member jitter in depth (4–6), subsample (0.7–1.0) and
colsample (0.65–1.0), plus two LR-diverse members. `train.py` prints `Eval AUC: 0.7432`;
`predict_proba(df)` reproduces it on raw frames without the target column (`CONTRACT OK`).

## Changes that mattered most

1. **carrier × 30-min-slot pair categorical** (+0.007 on top of carrier×hour, +0.013 for carrier×hour itself):
   delay propensity is driven by the carrier's schedule position in the daily cascade; 30-min
   granularity was the sweet spot (hour too coarse, 15-min too sparse).
2. **Frequency encodings** of Origin/Dest/Route (+0.003): log1p flight counts fit on train only —
   drift-robust proxies for airport busyness, unlike target encodings.
3. **Capacity retune** (~200→350 trees, depth 6→5, +0.004): shallow trees at moderate count
   generalize best across the 2005→2006 time shift; deep or long models overfit 2005.
4. **DepTime feature engineering** (+0.0003): hour/minute decomposition, cyclical sin/cos, log-distance.
5. **Diverse bagging** (5→12 members, +0.001): needs real param jitter — seed-only bags are
   identical without subsampling.

## Things that did not help

- **Target encoding** (Origin/Dest/Carrier/Route/hour, OOF + smoothed; also TE of carrier×hour):
  airport delay rates drift year-to-year; every TE variant scored below the categorical baseline.
- **More pair interactions** (Origin×hour, carrier×day-of-week, dow×slot30): all diluted the signal —
  only the carrier×time interaction family worked.
- **Stronger regularization / different boosters** (min_child_weight=10, gamma=1.0, DART members,
  early stopping + refit): ties or losses; DART also tripled runtime.

## With more budget

I would search the interaction granularity space more systematically (carrier×slot at 20/40/60-min
cuts, slot features keyed to scheduled block time rather than clock time), try a two-level stack
(XGBoost meta-learner on out-of-fold bag predictions), probe colsample_bylevel/bynode regularization
for the bag, and re-tune learning rate jointly with tree count per member instead of sharing one base.
