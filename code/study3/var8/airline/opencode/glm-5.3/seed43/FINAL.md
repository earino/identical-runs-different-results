# Final report — autoresearch XGBoost (airline delay, hidden-holdout scenario)

**Best Eval AUC: 0.7482** (experiment #11, commit 78a84da) — single lossguide XGBoost, lr 0.01, early stopping on **AUC** (es=200), 21 engineered features. Validation: `CONTRACT OK` (predict_proba reproduces 0.7482 with the target column removed).

## Changes that mattered most

1. **Time-of-day engineering** — minutes-since-midnight (`tod`) plus sin/cos of the daily cycle (0.7141 → 0.7192).
2. **Smoothed target encodings** for carrier/origin/dest (m=10) instead of raw categoricals/one-hots (→ 0.7224).
3. **Carrier×hour TE (m=30)** plus log flight-count features for origin/dest/route (→ 0.7282).
4. **Airport×hour congestion counts** (1-hour buckets, log1p) for origin and dest (→ 0.7435).
5. **Early stopping on AUC** (`eval_metric="auc"`) instead of the default logloss, with lr 0.01 and max_leaves 256 — the late-regime jump (0.7452 → 0.7482). Logloss-optimal stopping was cutting AUC-optimal training ~3000 trees short.

## Things that did not help

1. **Entity×time encodings beyond carrier×hour**: route TE, origin/dest×hour TE, origin×month TE, holiday-distance features — all overfit 2005 idiosyncrasies and transfer poorly to 2006.
2. **Ensembling under the time cap**: diverse 2-model blends (0.7471), 4-seed variants, dart — a single, better-stopped model beat all of them within 120 s.
3. **Ablations and exotic extras**: dropping minute/day/log_distance cost −0.0030; relative congestion shares, OOF target encoding, recency sample-weighting, monotone constraints, gamma/lambda/colsample variants — all equal or worse.

## With more budget

I would explore ensembling that fits the time cap via faster base learners (quantile DMatrix, smaller trees with lr≈0.02) so that AUC-ES members can be averaged instead of relying on a single model's iteration choice; a cross-validated stopping target (average best-iteration over 3 folds of 2005 to reduce eval-set dependence of the stopping point); finer time-resolution carrier×time encodings (the half-hour TE timed out at 120 s but the curve was still improving — worth a leaner retry); and calibration-aware blending of lossguide and depthwise members trained on disjoint feature subsets.

## Budget note

CPU budget (18,000 s) was the binding constraint: exhausted at experiment #18 of 40 (≈105 min wall clock used). Last run_experiment status: 22 experiments nominally left, CPU 17,646/18,000.
