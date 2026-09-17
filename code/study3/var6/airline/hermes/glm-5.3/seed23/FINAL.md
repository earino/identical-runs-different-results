# FINAL — airline dep-delay XGBoost

Best Eval AUC: **0.7575** (validated via ./validate.sh, CONTRACT OK; baseline was 0.7141, +0.0434).

Final design (train.py): an ensemble of 3 XGBoost models (max_depth=12, lr=0.03, 400 trees,
colsample_bytree=0.3, subsample=0.85) over features = raw columns + one-hot dates/airports +
cyclic hour/date encodings + smoothed out-of-fold target encodings of categorical x time-bin
interactions, each member built on a different 5-fold shuffle of the OOF TE.

## The changes that mattered most

1. **Out-of-fold target encodings of categorical x scheduled-time interactions** (fit on train only,
   OOF for train rows, full-train map applied inside prepare()). The single biggest lever: 0.7141 -> 0.73
   with 8 keys. The most valuable keys were route(origin x dest) x 20-min time bin, origin x 15-min bin,
   carrier x 20-min bin, plus distance-bin x time.
2. **Fine time resolution in the TE keys**: hourly -> 30-min -> 20/15-min bins was worth about +0.013
   alone (0.7306 -> 0.744). 20/15-min mixed per key was optimal; 10-min overfit.
3. **TE count companion features**: every TE ships with its training-set count so trees can gate trust
   by evidence size; removing counts cost ~0.001. Also, without the count the over-smoothed TE is
   unrecoverable and AUC drops.
4. **Deep, heavily column-subsampled members** (depth 12, colsample 0.3) — the late, largest jump:
   depth 6/cs 0.6 -> depth 12/cs 0.3 moved 0.7540 -> 0.7575. With one-hot + TE features, deep narrow
   trees generalize better here than shallow bushy ones.
5. **Ensembling across TE-fold shuffles** (different OOF fold seeds per member) instead of seed-only
   ensembles: decorrelates members through TE noise, +0.0005-0.001 per step, and the averaging
   smooths the fold-level TE noise for the hidden holdout.

## What did not help (kept out)

- Early stopping on a within-2005 split: stopped too late, picked configs that overfit 2005-specific
  route/month effects (worse than baseline depth-6 30-tree model when pushed to 2000 trees).
- Route as a raw categorical feature: -0.01. Route information only works as a smoothed TE.
- Seasonality TEs (route x month, carrier x month): no gain; month effects shift year over year.
- Day-of-week interaction TEs (route x dow, hour x dow, ...): AUC-neutral at best; dropped 5 keys.
- Congestion counts (airport x hour flight counts), log-distance, night flags, hub sizes: all <= +0.0003.
- Recency weighting of late-2005 rows, 10-fold OOF, dual smoothing, rank-averaging: all neutral or worse.
- Plain one-hot alone with deep trees: 0.709 — TEs carry the signal, not the raw categoricals.

## What I would try with more budget

The TE-key space is the highest-value direction but it saturates around 13 keys; the remaining wins came
from the model side, and the depth/colsample frontier (d12 cs0.3) was still improving when the CPU budget
ran out. Next: (a) push to depth 14-16 with colsample 0.2 and fewer trees, validated for time; (b) 5-7
members at d12 if runtime allows (members cost ~27s each; 3 was what fit under the 120s experiment cap
together with the deeper frontier exploration); (c) hierarchical TE backoff built as a feature (TE blended
with its parent's TE weighted by count) instead of separate columns; (d) per-airport scheduled-departure
density computed from the full 2005 schedule as a covariate; (e) monotone constraints on the TE features
(safe direction: higher historical route-delay-rate -> higher risk) to reduce variance on unseen routes.
