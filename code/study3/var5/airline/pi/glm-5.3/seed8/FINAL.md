# Final report — airline delay classification (XGBoost)

**Best Eval AUC: 0.7352** (experiment #39, commit 655661a; validated via `./validate.sh` → `CONTRACT OK`,
`predict_proba` reproduces 0.7352 on eval.csv with the target column removed). Baseline was 0.7141.

## The final model
- Features (all built inside the `predict_proba` code path, fit on the 2005 train only):
  parsed calendar ints (Month/DayofMonth/DayOfWeek, day-of-year), DepTime plus hour/minute/mins-of-day,
  Distance(+log), native categoricals for UniqueCarrier/Origin/Dest, and 8 out-of-fold smoothed target
  encodings — carrier×hour, origin×hour, dest×hour, route×hour, carrier×origin, carrier×dest, dow×hour,
  carrier×dow — plus train-frequency counts (congestion proxies).
- Ensemble: 2 DART (dropout) XGB models (150 trees, depth 8, lr 0.18) blended with 5 plain gbtree XGB
  models (800 trees, depth 6/7/8, lr 0.05), each with different seeds and colsample; darts carry weight 2.0
  in the probability average. Runs in ~75s.

## What mattered most
1. **Time-of-day feature engineering** (hour/minute/mins-of-day from DepTime, parsed c-N ints): +0.003 over raw-string baseline; hour-of-day is the dominant signal.
2. **Interaction target encodings keyed on hour** (origin×hour, carrier×hour, route×hour, carrier×origin/dest), out-of-fold on train only, smoothed toward the global prior: +0.003 cumulative.
3. **Diversified bagging of XGBoost models** (seed + depth + column-subsample spread, lr 0.05 × 800 trees): +0.010 cumulative over a single model; the year shift punishes single-model capacity but rewards averaging.
4. **DART members in the blend**: +0.002; and the key discovery that *two small* darts (150 trees) beat *one large* dart (350 trees) while costing 8× less wall time — dropout-averaged trees are the best decorrelated partner for a gbtree bag.
5. **Pruning redundant columns** (plain carrier/origin/dest/route/hour TEs that native categoricals already cover) and **conservative smoothing** (TE_SMOOTH=40): +0.0024 and +0.0004 — with column subsampling, a leaner feature pool is a real accuracy lever.

## What did not help
1. **Capacity on raw features**: 3000 trees w/ early stopping on a 2005 split (0.7107), 300×depth-8 (0.7081) — the 2005→2006 shift makes same-distribution early stopping actively misleading.
2. **High-cardinality native categoricals**: route (5k levels) as an explicit categorical cratered AUC (0.6994); fancy TE variants — hierarchical parent-shrunk encodings (0.7294), hour-neighborhood TEs (0.7319), plain re-added TEs, month TEs — all worse than plain global-prior-smoothed interaction TEs.
3. **Other ideas that failed**: rank:pairwise objective (0.6032 solo — degenerate with one giant group), max_bin=128, row-bagging members, gbtree subsample 0.7, min_child_weight grid, 4-dart or 7-gbtree blends (blend weight ~0.25 for darts is the sweet spot), skipping dropout steps in dart.

## With more budget
I would attack the wall-time ceiling that blocks everything else: dart models cost ~0.35s/tree because of
dropout refits, which caps the ensemble at 7 members within 120s. With a faster or looser limit I would
grow a 20-40 member dart+gbtree mix with per-member feature-subset views, and re-tune blend weights by
time-blocked validation (train on early-2005 months, validate on late-2005) instead of the single 2006
slice — to pick the shrinkage, depth and blend weight that minimize shift sensitivity rather than slice-1
AUC. I would also probe whether the sparse-cell structure of route×hour TE (mostly a "seen-in-2005"
indicator) can be replaced by an explicit route-frequency × hour interaction, and try 2-level stacked
blending with a small XGB meta-learner over member predictions.
