# Final report — airline departure-delay AUC

**Best Eval AUC: 0.7354** (experiment #9, commit `f786f1a`). Contract validated: `./validate.sh` → `CONTRACT OK`.

## Task framing
Predict `dep_delayed_15min` (balanced Y/N) from 8 raw columns, train = 2005 (100k), eval = 2006 (100k).
The eval set is a *later year*, so the real risk was overfitting the previous-year distribution rather
than the eval file itself. I used a within-2005 temporal split (months 1–10 train / 11–12 validation)
as an honest proxy before trusting any change.

## Changes that mattered most
1. **Deep, fully-grown trees instead of shallow boosting.** The baseline (30 trees, depth 6) underfits;
   temporal validation showed ~29 trees was its optimum and more boosting only overfit. Switching to
   fully-grown trees (depth unlimited) raised temporal val from 0.706 to 0.728 — the single biggest win.
2. **Random-forest mode via `num_parallel_tree`.** One boosting round with 600 parallel deep trees,
   `subsample`/`colsample_bytree` < 1, is both faster and stronger than many boosted rounds (0.730).
3. **Averaging two complementary forests** (300 trees each at `subsample=0.9/colsample=0.5` and
   `0.8/0.4`) for seed/sampling diversity → 0.7354 and a more stable probability estimate.
4. **Cyclical time-of-day features** (`dep_sin`/`dep_cos`, wrapping post-midnight red-eyes past 24h) and
   month sin/cos, which sharpen the dominant DepTime signal.
5. **Correct train-only encoding path:** all feature engineering and the categorical levels live inside
   `prepare(df)`, so the hidden-holdout scorer reproduces them exactly.

## Things that did NOT help
- **Target encoding** of carrier/origin/dest/route (0.7258) — 2005 delay propensities did not transfer to 2006.
- **`colsample_bynode`** (per-node feature sampling, sklearn-RF style): 0.7125, much worse and slower.
- **Adding boosted deep trees to the forest average** (0.7319) and **route as a categorical** (0.706) both hurt.
- Hyperparameter micro-tuning (`min_child_weight`, `colsample` extremes) was within seed noise (~0.002 AUC).

## What I would try with more budget
Push the forest much larger and average over many independent forests (e.g. 5–10 × 300 trees with varied
`subsample`/`colsample`), since diversity was consistently the trend that helped and eval tracked temporal
validation well. Beyond that, quantile/rank aggregation of the forest outputs, calibrated bagged
`grow_policy="lossguide"` trees, and a proper month-based walk-forward validation would be the next
directions; feature engineering beyond DepTime/route mostly hit a ceiling because deep trees already
recover the interactions.
