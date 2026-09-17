# FINAL — airline delay (XGBoost autoresearch)

**Best Eval AUC: 0.7516** (experiment #40, commit 41b73ca; baseline was 0.7141 → +0.0375).
Final model: single `XGBClassifier`, `tree_method=hist`, `grow_policy=lossguide`, `max_leaves=2560`,
`max_depth=0`, `min_child_weight=0.5`, `learning_rate=0.01`, `n_estimators=600`, `subsample=0.5`,
`colsample_bytree=0.4`, 27 engineered features, native categoricals enabled. Validated: `CONTRACT OK`,
`predict_proba` reproduces 0.7516 on eval with the target column removed.

## Changes that mattered most (in order of discovery)

1. **Traffic-volume features** (exp #13, +0.0027): log1p flight counts per Origin, Dest, route
   (Origin_Dest), carrier, hour, (origin,hour), (dest,hour) computed from the 2005 training slice only.
   Target-free and stable across the year shift — congestion proxies the trees can't build from raw
   categoricals alone.
2. **Lossguide trees with many leaves** (exp #14→#27, ≈+0.010): switching from depthwise d12 to
   `grow_policy="lossguide"` and scaling `max_leaves` 256→512→1024→2048→2560 while *lowering*
   `min_child_weight` 10→1→0.5. Flat-leaf trees with heavy row/col bagging generalize across the
   2005→2006 shift far better than shallow balanced trees.
3. **Multi-scale time-of-day encodings** (exp #38→#40, +0.0045): DepTime→minutes-of-day plus sin/cos at
   24 h, 12 h, 6 h, 4 h and 8 h periods (and fractional hour). The day's delay curve is a sum of
   broad + shift-scale bumps; several Fourier bases give the trees cheap split geometries for each.
   Ablating three "redundant" encodings cost −0.0097, proving redundancy is a feature, not a bug.
4. **Low learning rate + bagging as year-shift regularization** (exp #6→#9, +0.005 from the 0.7145
   plateau): lr 0.01 with 400–600 rounds, subsample 0.4–0.5, colsample 0.4. Every doubling of
   smoothness transferred better to 2006; high-lr/few-round and unbagged deep configs overfit 2005.
5. **Seed/config micro-ensembles** (exp #18/#22/#24, +0.001): averaging 2–3 individually strong members
   gave small consistent gains, but once members grew to 2560 leaves two fits no longer fit in the
   120 s cap, so the final model is a single best member.

## Things that did not help

1. **Route as a raw categorical** (−0.009): 5k route levels = pure 2005 memorization.
2. **Target encodings from 2005** (−0.010 even with k=100 smoothing): 2005 delay rates per
   carrier/route/hour simply don't rank 2006 flights.
3. **Day-of-week volume features, day-of-year cyclics, carrier-share ratios, dist-vs-route-mean**
   (−0.004 to 0): dilute the colsample budget without adding transferable signal; adding features
   under colsample 0.4 steals splits from the good ones.
4. Also negative: min_child_weight ≥ 10 at high leaves, λ/reg_lambda ≥ 3, subsample 0.6/col 0.5
   together (−0.006), early stopping on a random 2005 split (it stops at ~90 rounds and underfits),
   rank/lossguide-128 members in ensembles, r900 rounds.

## With more budget

The single biggest wall was the 120 s per-experiment cap colliding with 2560-leaf trees: an ensemble
of 3–5 of the final members (different seeds + leaves 2048/2560 + sub 0.4/0.5) would very likely add
+0.001–0.002 that I could not reach in-run, and a proper CPU-budgeted search over
(colsample, sub-sample)×(leaves, rounds) around the optimum would too. I'd also fit the Fourier basis
more systematically (sweep 2 h–12 h periods, learn amplitudes per Origin), try `max_bin=512` and
`refresh_leaf`/continuation boosting at the final scale, and test whether a second model trained only
on late-2005 months (recency weighting) transfers better to 2006. Finally, the volume-feature family
suggests modeling *schedule network structure* directly (e.g., per-airport hourly departure counts
normalized by airport size, plus arrival-side proxies) — target-free but distributional, which is
exactly what survives a year boundary.
