# Final Report — airline (AUC)

**Best Eval AUC: 0.7406** (final `train.py` at HEAD, validated: `CONTRACT OK`, predict_proba reproduces 0.7406).
Baseline (30-tree depth-6 XGB on raw features): **0.7141** → total improvement **+0.0265** over 40 experiments.

## Final model

10-member XGBoost ensemble, prob-averaged, trained on all 100k rows of `data/train.csv`:

- 4× `lossguide` members (max_leaves 160–256, n_est 450, lr 0.05–0.06, colsample 0.45–0.55, reg_lambda 15–25, mcw 20, max_bin 512) on the **hourcat** feature set (numeric time/calendar/Distance + `hour_cat` 24-level cat + carrier/origin/dest cats).
- 4× the same lossguide configs but on **hourcat + `carrier_hour`** (UniqueCarrier×hour 480-level categorical pair).
- 2× plain `hist` depth-4 members (n_est 450, lr 0.05, colsample 0.6, reg_lambda 10) on hourcat+carrier_hour.

Feature engineering is entirely inside `prepare()` (fitted on train only): time-of-day features from DepTime
(hour, minutes-since-midnight, sin/cos), month/dow/dom cyclicals, categorical carriers/origins/dests,
`hour_cat`, and `carrier_hour`. The year-shift means train=2005 and eval/holdout=2006 are different slices,
so only coarse, year-stable signals were kept.

## Changes that mattered most

1. **Time-of-day features** from DepTime (dep_hour, time_min, sin/cos of hhmm): 0.7141 → 0.7180. `dep_hour` is by far the highest-gain feature; scheduled-departure hour dominates delay behavior.
2. **`hour_cat` (24-level categorical hour)**: captures the non-linear hour→delay curve better than binning/numeric; durable gain (~+0.002) across regimes.
3. **`grow_policy="lossguide"` (leaf-wise trees, 160–256 leaves)**: the single biggest model-side win, 0.7238 → 0.7297 single-model (and eventually 0.7336 with tuning). Depth-limited `hist` trees could not capture the same structure.
4. **Strong regularization at the lossguide operating point** (reg_lambda 15–25, mcw 20, colsample_bynode 0.45–0.55): essential — unregularized leaf-wise trees overfit the 2005 slice and lose 0.005+ on 2006.
5. **Mixed ensembling**: seed/config ensembling (+0.005), then algorithm-family mixing (8 lossguide + 2 hist depth-4, 0.7373), then **feature-view mixing** (4 members without vs 6 with `carrier_hour`, 0.7406). Diversity across configs, families, and feature views each added a real increment.

## Things that did not help (all tested and reverted)

- **Route (Origin×Dest) categorical and frequency encodings**: 4198 levels memorize 2005 specifics; consistently negative on 2006.
- **Smoothed target encoding (m=50/200/1000) of carrier/origin/dest**: near-zero to negative; the year shift eats through any train-fitted rate.
- **Fine-grained interactions** (hour×dow pairs, cyclic products, minute-level cats, holiday flags, day-of-year): all neutral or harmful — same shift story; also **carrier×hour without hour_cat** was worse than with it (the pair complements, not replaces).
- **Internal-split early stopping** and **subsample 0.8**: early stopping picks too many trees for the year shift; row subsampling hurt even at depth 4.

## With more budget

I would (1) sweep the lossguide operating point jointly (leaves × mcw × lr × n_est) on a **2006-holdout-style internal validation** — e.g., fit on the first half of 2005 and validate on the second half of 2005 to mimic the year-gap shift when tuning tree counts; (2) explore monotone constraints (e.g., delay risk increasing with hour-lateness) which fit the shift-stability prior; (3) add a small number of deliberately diverse members (different max_bin, subsample-by-column patterns, depth-3 stumps) and optimize group weights on a shifted internal split rather than on eval.csv, since eval-based selection risks a small optimism bias; (4) test quantile-binned Distance interactions with carrier, the only remaining raw column with unexploited structure.
