# Final report — autoresearch XGBoost (airline, scenario 2)

**Best Eval AUC: 0.7219** (baseline 0.7141, +0.0078). Final commit `b6fb581`, validated with
`CONTRACT OK` (`validate.log`: AUC via `predict_proba` with target column removed = 0.7219).
Budget: 40/40 experiments, 213 min wall, ~1.9k of 18k CPU-seconds used.

## Final model

Heterogeneous XGBoost bag: 14 config families (depths 3/4/5/6/8/10/12 × {30 trees @ lr 0.1,
60 @ lr 0.05}) × 3 seeds = 42 members, `subsample=0.9, colsample_bytree=0.85`, hist trees with
native categoricals, probability mean. Features: raw DepTime (int), Distance, raw
UniqueCarrier/Origin/Dest/DayOfWeek categories, plus smoothed dep-delay-rate target encodings
("ddr", fitted on train only, smooth=50, unseen→prior) for Month, DayOfWeek, UniqueCarrier,
Origin, Dest, DayofMonth. Raw Month and DayofMonth categories are dropped — only their ddr
encodings remain. All engineering lives in `prepare(df)`; `predict_proba` re-applies it to
any raw frame, so the hidden holdout gets identical treatment.

## The 5 changes that mattered most

1. **Heterogeneous bagging of 42 small XGBoost models** (exp11→14→27→40): +0.008 of the total
   +0.0078. Single bigger/deeper models all *lost* to the 30-tree baseline; averaging small
   diverse members was the one lever that consistently paid. Wider depth range (3–10, then 12)
   helped twice (+0.0008 each); member count beyond ~40 and sampling twists did not.
2. **Dropping raw DayofMonth (exp34, +0.0015)** — 31 levels of pure memorization noise; keeping
   its smoothed ddr encoding instead.
3. **Dropping raw Month (exp36, +0.0001, simpler)** — same drift logic; month delay rates have
   only 0.59 correlation between 2005 and 2006.
4. **Smoothed ddr target encodings (exp15, +0.0003)** for the 6 categoricals, always fitted
   inside CV folds for validation and on train only for the final model.
5. **Internal CV diagnostics inside train.py** (exp7 onward) — not a direct AUC gain, but it
   exposed that in-sample CV gains did not transfer to eval, redirecting all effort to
   variance reduction and drift-robust encodings.

## 3 things that did not help

- **Bigger single models / more trees** (exp2–6): 200–2000 trees at depth 6–8 scored 0.6989–
  0.7132, all below the 30-tree baseline — the 2005→2006 shift punishes sharp in-sample fit.
- **Route-level target encoding** (exp26, 0.7082): route delay rates drift worse than carriers
  (corr 0.571 across years; LAX_LAS 0.60→0.43). Same for ddr of dep_hour (exp19) and volume
  count features (exp16).
- **Structural bag variants** (exp21/23/25/31): per-member feature subsets, row bootstrap,
  sampling/binning rotation, median blending — all below plain heterogeneous seed bags;
  decorrelation that drops signal costs more than the variance it removes. (Also: rank/logit
  averaging = plain mean; ddr smoothing 100 = 50.)

## With more budget

The next directions, in order: (1) even wider depth/config diversity in the bag (depths 2 and
14+15, lr 0.2 short members) since that axis kept paying through exp40; (2) a cheap
greedy forward selection of *which* ddr columns to keep per member family; (3) two-stage
XGBoost stacking on OOF predictions (untried — the only structurally new idea left); (4)
month-seasonality harmonics retested on top of the lean feature set, since the earlier test
(exp20) predates the raw-Month drop that changed the feature mix.
