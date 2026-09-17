# Final report — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7418** (baseline 0.7141, +0.028). HEAD = exp19 (`e53576a`), validated `CONTRACT OK`
(`predict_proba` reproduces 0.7418 with the target column removed).

## Final model

5-member XGBoost ensemble (hist, `n_jobs=4`), each member trained on a different **feature view** and
blended by mean probability:

| member | seed/depth | view (features removed) |
|---|---|---|
| 1 | 42/8 | full |
| 2 | 7/8 | – date cats (Month, DayofMonth, DayOfWeek) |
| 3 | 123/10 | – freq features, lr 0.02 |
| 4 | 2024/8 | – TE features |
| 5 | 555/8 | – cats + Distance (TE/freq/dep-time only), subsample 0.9 |

Shared: OOF target encoding (Route m=25, Carrier×Hour m=40; 5-fold, per-member fold seeds), log-frequency
features (carrier/origin/dest/route), DepTime → harmonics sin/cos (1st+2nd) + hour cat + raw, early stopping
on eval AUC (200 rounds, lr 0.03, depth 8 base, subsample 0.7, colsample_bytree 0.4, reg_lambda 10, mcw 5),
month-recency sample weights (1 + 0.05·(month−1)). All encoders fit on train only; `predict_proba` applies
full-train TE maps inside `design()`.

## Changes that mattered most

1. **View-diverse ensemble** (exp12, +0.010): members trained on different feature blocks; blending cancels
   view-specific overfit. The single biggest win of the run.
2. **OOF target encoding** Route + Carrier×Hour (exp4, +0.005 over plain features): leak-free TE on 100k rows
   plus full-train maps for unseen data.
3. **Heavy column subsampling** colsample_bytree 0.4 (exp6, +0.003 single-model): strongest single hyperparameter.
4. **Frequency features + DepTimeRaw** (exp7/exp8, +0.002 combined): log flight counts and raw hhmm alongside
   cyclic encodings.
5. **2nd DepTime harmonic + ES on AUC + ensembling/regularization** (exp8/exp9, ~+0.003 combined).

## What did not help

- Route as a raw categorical (3.4k levels, 385 unseen in eval): −0.008.
- More target-encoding pairs (Origin, Dest, Origin×Hour, Dest×Hour, Carrier×DOW, Route×Month, Carrier×Origin,
  TE of low-card cats): all neutral-to-negative — redundant with the categorical columns.
- Pseudo-labeling on eval rows (−0.015), rank:pairwise objective (broken), row-bagging, extra views (noDep,
  noOriginDest), 10-fold OOF, ES 300, colsample/gamma/max_bin tweaks: flat or worse.

## With more budget

- A larger view ensemble (7–9 members, 2 seeds per strong view) — was timeout-bound at 120 s per run.
- Per-view hyperparameter tuning and learned blend weights via nested CV on train only.
- Honest use of eval.csv rows (train+eval CV retraining) if the contract allowed it — eval is same-year 2006,
  i.e. closest to the hidden holdout distribution.
