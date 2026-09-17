# Final report — airline delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7588** (experiment #23, commit `44ad1e2`; validated end-to-end via `validate.sh` → `CONTRACT OK`).

Baseline (30 trees, depth 6, raw categoricals) scored **0.7141**; final model gains **+0.045**.

## Final configuration

3-seed ensemble (seeds 1, 2, 3) of XGBoost classifiers, probability-averaged:
`n_estimators=220, max_depth=24, learning_rate=0.0185, min_child_weight=1, subsample=1.0,
colsample_bytree=0.35, max_bin=512, tree_method=hist, enable_categorical=True`.

Features (all computed inside `prepare()`, all encoders/statistics fitted on 2005 train only):
integer Month/DayofMonth/DayOfWeek parsed from `c-<n>`, DepHour, DepMins (minutes of day),
**cyclic sin/cos of DepMins**, Distance + log(Distance), native categoricals for
UniqueCarrier/Origin/Dest, a **Hour x Carrier interaction categorical**, and 2005-fitted
**busyness counts**: flights/day at (Origin, hour), (Dest, hour), (Carrier, hour), plus each
Origin/Dest/Carrier's total daily volume.

## The 5 changes that mattered most

1. **Time-of-day feature engineering** (+0.004 alone, but enabling for everything else): parse the
   `c-<n>` strings to ints, extract hour/minute-of-day from DepTime, add cyclic sin/cos. Sincos
   turned out to be load-bearing: removing it later cost −0.008.
2. **Hour x Carrier interaction categorical** (biggest single lever, ~+0.005): evening-delay
   propensity differs by carrier; a categorical lets deep trees carve it directly.
3. **Deep, heavily column-subsampled trees with few boosting rounds**: depth 20→24, colsample 0.35–0.4,
   no row subsampling, lr ~0.02, only ~200-230 trees. In this regime capacity is spent on the
   systematic (transferring) structure while colsample decorrelates and prevents 2005 memorization.
   Naive scale-up (depth 8, ES) had *hurt* (0.708); the tuned deep regime reached ~0.75.
4. **Busyness count features** (+0.003): average daily flights at (Origin/Dest/Carrier, hour) and
   total daily volume, fitted on 2005. Airport/carrier schedule structure is stable year-over-year,
   so congestion proxies transfer to 2006.
5. **3-seed ensemble + binning/colsampling polish**: averaging 3 seeds (member eval spread was
   ±0.002) and max_bin=512 / colsample 0.35 / n220-lr0.0185 / depth 24 each added a few tenths of
   a millipoint (0.7542 → 0.7588).

## 3 things that did not help

1. **Anything that indexes 2005-specific noise**: Origin-Dest route categorical (4198 levels,
   −0.008), route/airport TE (−0.004 to −0.01 even with k=150 smoothing), day-of-year
   (Month x DayofMonth) categorical (−0.02), MonthHour/DowHour/CarrierDow interactions — all hurt
   because their statistics are 2005-particular and don't transfer to 2006.
2. **Early stopping on an internal 2005 split** (logloss or AUC): the 2005→2006 distribution shift
   makes in-year validation overtrain (best_iter ≈ 2500 → 0.7173 vs 0.75+ for fixed tree counts).
3. **DART, ordinal (label) encoding instead of native categoricals (−0.005), per-node colsample
   (−0.018), gamma/alpha regularization, weekly-cycle sincos, holiday calendar flags, hub-share
   ratios, finer carrier×30-min blocks, config-diverse ensembles, and 4-5 member ensembles of
   weaker members**: all neutral-to-worse.

## What I would try with more budget

The model is at a stable plateau (~0.759 on eval, all recent tweaks within ±0.001, mostly noise).
The binding constraint is the 120s experiment limit: 3 deep members cost ~107s, so ensemble size is
capped at 3. With more budget/compute I would (a) train 10-20 members and average, which by the
measured seed spread should add ~+0.001-0.002 and, more importantly, make the holdout score
less seed-dependent; (b) explore a proper treatment of the year shift — e.g., fitting on
month-blocked folds of 2005 and validating on the *latest* months, or adversarial-style weighting
that downweights features/regions of the 2005 sample that don't re-occur in 2006; (c) revisit
route-level information with hierarchical shrinkage (airport-pair effects shrunk toward
origin+dest marginal effects) rather than raw categories or flat TE, since route identity is
clearly informative but too sparse to estimate from 100k rows; and (d) sweep the depth/colsample
frontier more finely with multi-seed evaluation (single-seed eval diffs proved unreliable at the
±0.002 level — several "improvements" were seed luck, and my keep/discard decisions on sub-milli
gains should be read as approximate).
