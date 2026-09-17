# Final report — airline delay XGBoost (autoresearch harness)

**Best Eval AUC: 0.7223** (baseline: 0.7141, +0.0082). Final `train.py` = HEAD, validated (`CONTRACT OK`,
AUC reproduced via `predict_proba` with the target column removed).

## Final model

Ensemble of 3 diverse XGBoost classifiers (probability-averaged), each trained on the full 100k-row
`data/train.csv` with leak-free engineered features:

- Members: (max_depth 5, colsample 0.7), (4, 0.8), (6, 0.6); shared: lr 0.03, n_estimators 600,
  min_child_weight 12, gamma 1.0, subsample 0.8, `tree_method=hist`, `enable_categorical=True`.
- Features: raw DepTime + hour + minute-of-day, Distance + log1p(Distance), log traffic counts
  (route, origin×hour, dest×hour); categoricals Month / DayofMonth / DayOfWeek / UniqueCarrier /
  Origin / Dest (train-fitted levels, unseen → NaN); **10 smoothed target encodings** — carrier,
  origin, dest, route, month, day-of-week, hour, origin×hour, dest×hour, carrier×hour — fit on train
  only, 5-fold OOF values for training rows, full-train maps applied inside `prepare()` at predict time.

## What mattered most (3–5 changes)

1. **Smoothed target encodings** (train-fit, 5-fold OOF on train rows) — biggest single lever once the
   model had capacity; interaction TEs with hour were the strongest (eval 0.7156 → 0.7190 at equal config).
2. **Diagnosing the 2005→2006 distribution shift with learning curves** — val (2005) AUC rises while eval
   (2006) AUC falls past ~100–300 trees; this redirected all tuning toward *lower-complexity, better-
   regularized* models instead of more trees.
3. **Regularization + capacity balance**: depth 5 (better than 6 and 8), colsample 0.7, mcw 12, gamma 1.0,
   lr 0.03 with ~600 trees → 0.7202–0.7209 single-model.
4. **Diverse 3-member ensemble** (depth × colsample diversity beats same-config seed averaging:
   0.7221 vs 0.7210): 0.7223 final.
5. **Keeping raw Origin/Dest categoricals alongside TE** — dropping them cost −0.002; they carry signal
   beyond smoothed encodings.

## What did not help

- **Deeper/wider early-stopped models** (depth 8, ES on a 2005 split): eval AUC *fell* with more trees
  (0.712 peak at 60 iters vs 0.7141 for the tiny baseline) — temporal overfitting.
- **Extra interaction TEs** (dow×hour, carrier×month, origin×dow) and **calendar-position TE**
  (Month×DayofMonth): 0.7169 / 0.7117 — day-specific and seasonal-carrier effects don't transfer across years.
- **Bootstrap bagging** (0.7188) and **more ensemble members** (6-member 0.7214, 4-member with lr diversity
  0.7217): members lose too much unique data / the good diversity mix got diluted.

## With more budget

I would (a) build a small stacked model — XGBoost meta-learner over out-of-fold predictions of several
diverse members plus a few base features; (b) explore per-feature sampling weights to upweight the
year-stable signals (hour, day-of-week, carrier) relative to noisy airport-specific splits; (c) tune TE
smoothing per key by maximizing *stability across artificial year-halves of 2005* rather than eval AUC
directly, which should generalize better to the hidden 2006-slice-2 holdout; (d) probe DepTime
minute-level and congestion-style features more systematically.
