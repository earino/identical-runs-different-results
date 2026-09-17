# Final report — airline dep_delayed_15min, XGBoost (autoresearch harness)

**Best Eval AUC: 0.7355** (baseline 0.7141, +0.0214). Best commit `d08c590` ("dest*dist3 TE");
validation passed (`CONTRACT OK`, `predict_proba` reproduces 0.7355 on eval with the target column removed).

## Final model (train.py at HEAD)

- Features: raw DepTime + cyclic time-of-day/day-of-year, calendar ints, Distance/log-distance,
  estimated arrival time-of-day (`tod + Distance/500 + 0.5`), overnight flag, 3 raw categoricals
  (XGBoost native), and 30 smoothed target encodings (TEs) built out-of-fold (5-fold) for training rows
  and from full-train stats for unseen rows. TE families: entity (carrier/origin/dest/route),
  hour-of-day x entity, day-of-week x entity, 30/15/7.5/5-min departure buckets (tod48/96/192) x carrier,
  arrival-bucket x carrier, tod48 x dow, dest x 6-h period, origin/dest x distance tercile, tod48 x distance.
- Ensemble: 16 XGBClassifiers = 2 feature views x 8 param configs (depth 5-8, colsample/subsample 0.7-0.9),
  probability-averaged; 200 trees @ lr 0.05, min_child_weight 5, hist, 4 threads. Trains in ~25 s.

## Changes that mattered most

1. **Kill early stopping; fixed 200 rounds @ lr 0.05** (+0.004): 2005 validation AUC rises monotonically
   while 2006 eval AUC peaks around 200-300 rounds — early stopping systematically overshot.
2. **Out-of-fold, smoothing-shrunk target encodings of interactions** (+0.007 over the bagged base):
   hour x carrier/origin/dest, dow x carrier/origin, route. Single biggest early lever (te_hour_carrier
   is the top feature, ~0.23 gain importance).
3. **Fine-grained time-bucket TEs** (+0.008 cumulative): 48 -> 96 -> 192 departure-time buckets with
   carrier interactions, plus an arrival-time proxy (from Distance) with its own bucket x carrier TEs.
4. **Ensemble diversity** (+0.002, then +0.002): 8 param-diverse configs beat same-config seed bagging;
   then doubling as two *feature views* — 8 members with raw categoricals + 8 members with categoricals
   dropped (TE numerics only) — gave the largest single late gain (+0.0017).
5. **Distance- and dow-interaction TEs** (+0.0003): origin/dest x distance tercile, tod48 x dow,
   dest x 6-hour period — small but robust.

## Things that did not help (all reverted)

1. **Early stopping in any form** (logloss or AUC on a 2005 holdout split) — see above; the 2005->2006
   shift breaks round-count selection. Also 3000-round lr-0.05 configs were worse.
2. **Calendar/seasonal features**: origin/dest/carrier x month TEs (-0.0036!), holiday proximity (-0.001),
   doy cyclicals neutral — monthly patterns from 2005 do not transfer to 2006.
3. **Volume/congestion features** (departures per origin/hour window), DART booster (-0.003 single-model),
   10-fold OOF (-0.0001), 32-member same-view ensemble (-0.0001), fold-complement data partitions (-0.0002),
   a third feature view (-0.0002), rank-averaging (equal), dropping raw DepTime (equal).

## What I would try with more budget

With the time-bucket TE vein plateauing (192 buckets is the finest support allows) and ensemble
diversity exhausted at the param level, the promising directions are: (1) proper stacking — OOF
predictions from view-submodels as meta-features of a small logistic/xgb meta-learner, which could
beat uniform averaging if views have complementary error structure; (2) empirical-Bayes TEs with
per-key posterior variance (uncertainty-aware encoding) instead of fixed smoothing constants; (3)
year-robust calendar encoding (week-of-year with strong shrink, or interactions of season with
tod-buckets to see which part of the day-profile actually shifts between years); (4) more feature
views (drop all entity TEs / drop all time TEs) with per-view blend weights tuned by 2005-fold CV
validated against the 2006 shift rather than by eval AUC, to keep keep/discard decisions honest.

Budget used: 40/40 experiments, ~168/230 min wall, ~3900/18000 CPU-s. 27 kept improvements,
13 reverted (incl. 2 crashes: a dict-indexing bug in the first OOF-TE attempt, and an indexing
bug in the last fold-complement experiment whose fix landed after the counter ran out).
