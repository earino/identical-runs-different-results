# Final report — airline delay XGBoost (autoresearch harness)

**Best Eval AUC: 0.7278** (exp37, commit b07646d). Baseline was 0.7141 (+0.0137).
Final model: an 10-member XGBoost ensemble (5 diverse configs × 2 seeds), mean of predicted
probabilities, all trained on `data/train.csv` with engineered features; `predict_proba(df)`
re-derives every feature from a raw frame (contract verified by `validate.sh`).

## Changes that mattered most

1. **Time features from DepTime** (exp3, 0.7141→0.7204): parse `c-<n>` strings to ints for
   Month/DayofMonth/DayOfWeek; split DepTime into numeric `hour`, `minute`; native categoricals
   for carrier/origin/dest; deeper model (800 trees, lr 0.05, depth 8) instead of 30×depth-6.
   Time-of-day is the dominant signal (delay rate 4%→83% across the day).
2. **Config-diverse seed bagging** (exp11–15, →0.7259): average of 5 configs × 2 seeds —
   depths 6/8/10/12 (with varied subsample) plus a `lossguide, max_leaves=64` config.
   Diversity beat same-config seeds (5×d8 = 0.7221 < diverse 10 = 0.7259); even weak members
   (d12) helped by decorrelating the average.
3. **`dt5` = DepTime//5** 5-minute-bin feature (exp23, →0.7267): raised every member's solo AUC
   by ~+0.005–0.010; scheduled-departure granularity carries real operational signal.
4. **Smoothed target-encoding interactions as single numeric columns** (exp29–32, →0.7274):
   carrier×hour (m=50), dow×hour (m=100), carrier×dow (m=30), origin×hour (m=500) — fit on
   train only, unseen keys → prior. Small but consistent gains.
5. **Route count encoding** (exp37, →0.7278): log1p count of the Origin|Dest pair (fit on train),
   unseen routes → NaN (XGBoost missing-path). Frequency is stable across years, unlike rates.

## What did not help

- **More capacity / lower learning rate** (exp5: 3000 trees lr 0.03): internal AUC rose to 0.7925
  but eval fell to 0.7181 — the 2005→2006 year shift punishes overfitting to 2005 patterns.
- **Target encoding of origin/dest/route + route categorical** (exp4: 0.7091): in-sample TE on the
  training matrix (even OOF-corrected) plus a ~10k-level route category overfit badly.
- **Categorical versions of month/dom/dow/hour** (exp7: 0.7154) and **interaction categoricals**
  carrier×hour / dow×hour / origin×hour (exp8: 0.7196): native categorical splits on discrete
  codes transfer worse than smooth numeric encodings.
- **Leaf regularization** (min_child_weight=20, reg_lambda=5 → 0.7199), **max_bin=64** (0.7243),
  **colsample_bynode=0.8** (0.7203), **logit-mean instead of mean-probability combining**
  (0.7271), **te_month_dom + distance residual** (0.7169): all neutral-to-negative.
- **Reallocating seeds to the strongest members** (exp17, exp34, exp36, exp40): consistently worse
  than balanced config diversity; individual member strength does not predict ensemble contribution.

## With more budget

Given the year-shift pattern, I would: (a) build a proper time-based validation protocol
(2005-holdout → 2006) to select hyperparameters without touching eval.csv; (b) explore
per-config early stopping on a 2005 tail split with year-robust features only; (c) try
monotone constraints on the time-of-day features; (d) grow the ensemble with cheaper members
(lossguide variants at 32/96 leaves, subsample jitter) under a shared-DMatrix `xgb.train`
refactor to stay inside the 120 s limit; (e) test whether dropping the carrier main features
(they may drift between years) in favor of carrier×hour smoothing improves transfer.
