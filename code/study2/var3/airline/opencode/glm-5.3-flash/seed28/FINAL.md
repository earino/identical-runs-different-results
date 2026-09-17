# FINAL — airline / XGBoost autoresearch

**Best Eval AUC: 0.7245** (commit `92124ac`, "staggered caps 800-1600 under ES").

## Final model

Ensemble of 5 XGBoost classifiers (depth 8, lr 0.06, subsample 0.8, colsample 0.6, reg_lambda 2,
min_child_weight 5, hist, native categoricals), each trained on a **different random 90/10 split** of
train.csv with **early stopping (60) on its own 10% valid** and a **staggered tree cap**
(800/1000/1200/1400/1600); predictions averaged.

## Changes that mattered most

1. **Bagged ensemble over data splits (5 seeds × own 90/10 split + ES)**: 0.7148 → 0.7244 (+0.0096).
   By far the biggest single win; split diversity — not seed diversity — is the engine (a common-split,
   hill-climbed-weight variant scored only 0.7184).
2. **Core feature engineering**: c-N dates parsed to ints, DepTime → minutes-of-day + sin/cos + hour,
   log1p(Distance), Origin/Dest/UniqueCarrier as native categoricals, Route (Origin_Dest) category:
   baseline 0.7141 → 0.7148.
3. **Right-sized capacity/regularization**: depth 8 + min_child_weight 5 + subsample/colsample ~0.7.
   Depth 10, depth 7, mcw 3/10, stronger or weaker subsampling, gamma, lower lambda, lr 0.07 all scored worse.
4. **Staggered tree caps under ES** (800–1600): +0.0001 — the final nudge to 0.7245.

## What did not help

- **Target encoding** (smoothed, and even leak-free 5-fold OOF): 0.7075 / 0.7119 — 2005→2006 shift makes
  historical group delay-rates unreliable.
- **Interaction features** (Month×DOM, Hour×DOW, Hour×Carrier cats, DistRel): 0.7043–0.7204 — every feature
  addition beyond the base set hurt; the simple feature set is a sharp optimum.
- **Objective/structure changes**: rank:pairwise members 0.7192, lossguide 0.7222, DART (too slow, timeout),
  full-data fixed-tree members 0.7219, 10-member/short-cap variants ≤0.7239, median combiner 0.7228.

## With more budget

- 15–25 members of the winning protocol (needs parallel fits or a faster per-member recipe to fit the 120 s
  limit) plus a small grid on ES patience.
- Per-member hyperparameter jitter (depth 8±1, colsample 0.5–0.7) at fixed protocol, judged on a *train-internal*
  CV score rather than eval.csv to avoid overfitting the keep/discard decisions.
- A second-year calibration check: recompute encoders on train+eval-like data if the task ever allows it.
