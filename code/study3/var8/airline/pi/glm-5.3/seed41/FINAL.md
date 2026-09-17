# Final report — airline XGBoost autoresearch

**Best Eval AUC: 0.7626** (experiment #33, commit `f61b253`; baseline was 0.7141, +0.0485).
Final model: 3-model bagged ensemble of deep, L1-regularized XGBoost trees; contract validated
(`CONTRACT OK`, predict_proba reproduces 0.7626 on a raw target-dropped DataFrame).

## The 5 changes that mattered most

1. **Deep trees + heavy stochastic regularization** (max_depth 24, subsample 0.75, colsample_bytree 0.5,
   reg_alpha 2): alone took the single-model score from 0.714 to 0.745. Unregularized deep/many trees
   *hurt* (more trees after 30 → steadily worse); row/column sampling + L1 turned depth into a win.
2. **DepTime decomposition + cyclical harmonics**: hour, minutes-of-day, sin/cos of time-of-day and its
   2nd/3rd/4th harmonics, plus |minutes-from-noon|. The 2nd–4th harmonics alone added ~+0.004 total
   (0.7580 → 0.7626); the time-of-day shape is the single dominant signal and harmonic terms let one
   split express multi-lobed time bands.
3. **Dropping drift-prone calendar features** (Month, then DayofMonth): exact-month/day patterns learned
   on 2005 don't transfer to 2006; dropping Month +0.001, DayofMonth +0.0026. DayOfWeek (numeric) stays.
4. **HourCat + stable count features**: departure hour as a native categorical (+0.005), plus
   (Origin×hour), (Dest×hour), (Carrier×hour), carrier size and route size flight counts fit on train
   only (+0.004 cumulative). Congestion proxies that recur year-over-year transfer; target encodings
   do not.
5. **Small seed-bagged ensemble + low learning rate** (3×400 trees at lr 0.021): +0.003 over a single
   model; kept wall time ~102 s, comfortably inside the 120 s cap (an n=520×3 variant timed out).

## What did not help (tried and reverted)

- **Route identity in any form**: raw Route categorical, route target-encoding, route-median-distance
  deviation — all negative (up to −0.008); the route label is a memorization trap across years, while
  route *size* (count) helps.
- **Target encodings** (Origin/Dest/Carrier/Route/hour): −0.001 to −0.008; 2005 label statistics shift
  and the native categoricals already carry the identity.
- **Fine calendar / derived time**: day-of-year, Month sin/cos, DOW/DayofMonth/quarter-hour as
  categoricals, weekly cycles, evening-peak distance, 5th harmonic — all neutral or worse.
- **Alternatives**: lossguide growth, colsample_bynode, rank:pairwise objective, training on recent
  months only, recency weighting, max_bin=128 — all worse or not better.

## With more budget

I'd (a) attack the ensemble-vs-120 s wall by shrinking predict cost (fewer, shallower trees with more
members or model compression) to fit 8–10 bagged members, worth maybe +0.002; (b) use a within-2005
temporal split (train Jan–Sep, validate Oct–Dec) as a second, independent validation axis for
feature-keep decisions — several keep/revert calls rested on a single 100k eval slice; (c) explore
semi-supervised uses of the 2006 structure (e.g. schedule-shape stability scores per airport/hour
measured as train-vs-holdout consistency) that don't touch labels; and (d) tune the
harmonic/abs-noon time family jointly with HourCat, since their gains suggest the time-of-day response
surface is still not saturated.
