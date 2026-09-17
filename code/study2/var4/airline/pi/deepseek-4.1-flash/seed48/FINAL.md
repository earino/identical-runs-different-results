# FINAL — airline XGBoost autoresearch

**Best Eval AUC: 0.7307** (commit `11ab200`, experiment #38: 4-seed XGBoost ensemble,
depth 8, lr 0.01, 3000 trees per model, `colsample_bytree=0.3`). Baseline was 0.7141.

## Changes that mattered most

1. **Heavy column subsampling (`colsample_bytree=0.3`)** — the single biggest win
   (0.7165 → 0.7267 at depth 6). With only ~11 usable features, most are noisy/collinear
   (`DepTime`, `tod`, `dep_hour`, `sin/cos_tod`); forcing each tree to use ~3 of them
   decorrelates the ensemble and massively improves year-to-year generalization.
2. **Low learning rate with many trees (`lr=0.01`, 5000+ trees)** — smooth additive
   fitting transferred better across the 2005→2006 split than the default `lr=0.1`/30 trees
   (0.7141 → 0.7181 before regularization was added).
3. **Time-of-day features** (`tod = hour*60+min`, `dep_hour`, `sin/cos` of `tod`). Crucially,
   `tod` gives a clean monotone clock axis whereas raw `DepTime` has the `hh59→hh00` jump.
   Removing these features collapsed AUC to 0.7175 (experiment #30).
4. **Deeper trees once regularized (`max_depth=8`)** — capacity only helped after column
   subsampling was in place (0.7267 → 0.7285); depth 10 was slightly worse, depth ≤6 left
   interactions on the table.
5. **Seed ensembling** (4 models × 3000 trees, averaged probabilities) — because column
   subsampling makes each tree very stochastic, averaging seeds reduced variance
   (0.7285 → 0.7307). It is also the most robust change for the hidden holdout.

## Changes that did NOT help

1. **Route features** — `Origin_Dest` as an XGBoost categorical and a smoothed target
   encoding of it both lost ~0.005–0.01 AUC. Route-level delay rates and even route identity
   do not transfer from 2005 to 2006 (385 eval routes are unseen), so they only add
   year-specific overfitting.
2. **Row subsampling (`subsample=0.8`)** and `max_bin=512` — both neutral-to-negative once
   column subsampling was in place; `subsample` cost 0.003.
3. **Numeric/cyclical calendar features** (weekend flag, `sin/cos` of day-of-week,
   replacing categorical Month/DayOfWeek/DayofMonth with numbers) — dropped AUC by ~0.004;
   the tree already handles the small categorical calendars well.

Also rejected: `max_depth` > 8 or < 6, `min_child_weight=10` + `reg_lambda=2`,
`colsample_bylevel`, `colsample_bytree` outside 0.3 (0.2→0.7248, 0.4→0.7271),
`lr=0.005`/12000 trees (worse than `lr=0.01`/5000), and single-model variants.

## What I would try with more budget

The dominant lever was regularization, not raw feature count, which suggests the ceiling on
this feature set is near 0.73. With more budget I would (a) test a genuinely diverse ensemble
— bagging over feature subsets and hyperparameter seeds, 10–20 members, using `xgb.train` and
out-of-core prediction to stay inside the time cap — and (b) build a proper *time-based*
validation split inside the 2005 training year (e.g. hold out the last months) to select the
number of trees/early stopping honestly, instead of selecting on `eval.csv`, which is only
100k rows and makes sub-0.0005 differences noise. On the feature side I would revisit
smoothed target encodings only with out-of-fold fitting and a validation split, and explore
airport-by-hour congestion proxies and departure-time interactions, since the departure clock
axis is clearly the strongest signal and is currently only used additively.
