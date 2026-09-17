# Final report — airline delay AUC

**Best Eval AUC: 0.7526** (baseline 0.7141 → +0.0385). Final commit `672c35a`, contract validated (`CONTRACT OK`, AUC reproduced via `predict_proba` on eval.csv with the target column removed).

## Final model
Rank-averaged ensemble of **8 XGBoost models** (seeds 44–51), each: `max_depth=20, learning_rate=0.04, subsample=0.8, colsample_bytree=0.45, min_child_weight=1`, hist tree method, native categorical support for UniqueCarrier/Origin/Dest, early stopping (50) on eval.csv (the 2006 slice — best available proxy for the hidden 2006 holdout).

## Changes that mattered most
1. **Feature engineering inside `prepare()`** (exp 2→4): DepTime decomposed into fractional hour + sin/cos cyclic encodings + red-eye flag; `c-<n>` strings mapped to ordinal numbers; log-distance. Kept Month/DayOfMonth/DayOfWeek as plain ordinal numerics.
2. **Dropping the Origin→Dest Route categorical** (exp 4, +0.0075): native-categorical route with thousands of rare levels overfit 2005 catastrophically and choked early stopping at ~30–50 trees; without it training ran to 300+ trees.
3. **Deep trees in a low-leaf-regularization regime** (exp 5–8, 27–33, +0.02 total): depth sweep found 16–20 optimal, and once combined with column subsampling, `min_child_weight=1` (no leaf regularization) was best — depth itself regularizes.
4. **Heavy bagging** (exp 16–17, +0.002): `subsample=0.7–0.8, colsample_bytree=0.45–0.6` beat full-data fits; the tuned single model reached 0.7457.
5. **Seed ensemble of the champion config** (exp 36–39, +0.0069): 8 seeds of the same config, rank-averaged → 0.7526. Diversity of seeds beat diversity of depths/hyperparams.

## Things that did not help
- **Target encoding** (out-of-fold, smoothed, carrier/origin/dest/route): −0.005; native categoricals for those columns were strictly better.
- **Frequency/count features** and **holiday-period flags**: both slightly negative (−0.001 to −0.002).
- **Native categorical date/hour features** (Month/Dom/Dow/DepHour as categories): −0.014 vs ordinal numerics; ordinal splits generalize across the 2005→2006 shift.
- Larger regularizers (`min_child_weight≥10`, `reg_lambda=0.1`), depth ≥ 24, `max_bin=512`: all neutral-to-negative.

## What I would try with more budget
- Scale the seed ensemble to 15–20 members with a hard `n_estimators` cap to stay under the 120 s limit (the 9-seed run timed out at 121 s; per-member cost is the binding constraint).
- Tune the champion config jointly with lr (0.03 vs 0.04) and per-member subsample jitter (0.75–0.85) for cheaper diversity than extra seeds.
- A small DART member or per-member feature drops for decorrelation, plus a proper OOF-stacked XGB meta-learner (infeasible here: 5-fold × 8 members exceeded the per-run time budget).
