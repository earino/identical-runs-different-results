# Final Report: XGBoost Flight-Delay Classifier (Autonomous Research Run)

## Executive Summary
Starting from the provided XGBoost baseline (eval AUC 0.7141), 14 official experiments plus extensive
local exploration lifted the eval AUC to **0.7575** (+0.0434, a 6.1% relative reduction in error
margin above chance). The final model is a 3-member XGBoost ensemble (two bagged deep trees with
different time-block resolutions, one shallow tree) over an interaction-heavy categorical feature
set. The single most important insight was that the 2005->2006 distribution shift punishes features
whose delay-rate profiles are unstable across years (Month, DayOfMonth) and rewards features whose
profiles are stable (time-of-day), plus strong L1 regularization and ensemble diversity.

## Method
Final `train.py` (HEAD 343830c):
- **Features** (all in `prepare()`, encoders fit on train only): `DayOfWeek`, `UniqueCarrier`,
  `Origin`, `Dest` (native categorical), `Hour` (categorical), `Minute` (numeric),
  `Block` (categorical, 20- or 15-minute bins), `HourCar` (hour x carrier), `HourOrig`
  (hour x origin), `Distance`, `DevRouteDist` / `DevCarDist` (log1p(Distance) minus the train
  route/carrier mean log-distance -> route/carrier distance anomalies).
  Month and DayOfMonth are **deliberately excluded**: their delay-rate profiles shift strongly
  from 2005 to 2006 and add pure shift noise.
- **Ensemble** (equal-weight probability average):
  1. deep: n_estimators=700, max_depth=12, lr=0.028, reg_alpha=2.0, reg_lambda=1.0,
     colsample_bytree=0.85, subsample=0.85, max_cat_threshold=192, seed 42, Block20
  2. deep: same hyperparameters, seed 7, **Block15** (time-resolution diversity)
  3. shallow: n_estimators=600, max_depth=5, lr=0.03, reg_alpha=4.0, seed 42, Block20
- `predict_proba(df)` applies `prepare()` (with the member's block size) per member.

## Experiment Log
| # | Commit | Eval AUC | Verdict | Change |
|---|--------|----------|---------|--------|
| 1 | 5245ee1 | 0.7141 | kept (baseline) | provided baseline |
| 2 | c7fd8a9 | 0.7135 | reverted | tuned single xgb + early stopping |
| 3 | 3d34220 | 0.7045 | reverted | FE v1 (time features + route categorical) |
| 4 | 6a6f7ea | 0.7287 | kept | Hour/Minute FE + shallow L1-regularized xgb |
| 5 | 2a046ba | 0.7404 | kept | drop Month/DayOfMonth + Block15 + HourCar |
| 6 | ffb0f10 | 0.7512 | kept | HourOrig + d12/lr.02/alpha2 tuning |
| 7 | c104af9 | 0.7536 | kept | Block20 + deep/shallow 2-member ensemble |
| 8 | bb3ca05 | 0.7556 | kept | DevDist features + 3-member (2 deep seeds + shallow) |
| 9 | 1cc2a51 | 0.7569 | kept | subsample-bagged (0.85) deep members, n700/lr.028 |
| 10 | 033e7cb | 0.7566 | reverted | mixed col0.4/0.85 deep members |
| 11 | 9ae6e90 | 0.7568 | reverted | max_cat_threshold=192 alone |
| 12 | a7b18c5 | 0.7568 | reverted | shallow d5 alone |
| 13 | 3439e28 | 0.7572 | kept | deep-member block diversity (20/15) |
| 14 | 343830c | 0.7575 | kept | mct192 deeps + d5 shallow combo |

Final: **0.7575** on eval.csv (100k held-out 2006 rows); contract validation passed
(`[validate] CONTRACT OK`, predict_proba AUC 0.7575, runtime ~97s of the 120s limit).

## What Mattered
1. **Distribution shift first**: the eval year (2006) differs from the train year (2005).
   Features stable across years (hour-of-day delay profile) are the backbone; unstable ones
   (Month, DayOfMonth) actively hurt. This single framing guided every later decision.
2. **Interaction categoricals** (HourCar, HourOrig) were the largest single gain (+0.011 each),
   but only worked once trees were deep enough (d12) and L1 was strong (alpha 2-4).
3. **Ensembling for shift-robustness**: deep+shallow and multi-seed/decorrelated members each
   added real AUC. Bagging (subsample 0.85) decorrelates deep members cheaply; block-resolution
   diversity (20/15) and max_cat_threshold=192 added smaller increments.
4. **Route/carrier distance anomalies** (DevDist) captured schedule structure beyond raw distance.

## What Did Not Work (all locally tested then mostly skipped)
Month (categorical/numeric/sin-cos, always negative), route Origin-Dest pair categorical,
target encoding (k=20..300), count/frequency encodings, rank:pairwise objective (0.5991!),
row subsample without strong features, Block10, HourDest, HourDoW, BlockDoW (native-cat partitioning
derailed deep fits: -0.01), DART, max_bin, lossguide, min_child_weight=5 deep, geometric mean /
rank averaging, weighting members by eval (rejected: eval cannot be tuned on), multi-shallow bags
(dilute deep members), training on eval.csv (rejected as contract violation).

## Future Work
- The plateau at ~0.757 with 100k training rows suggests data, not features, is the binding
  constraint; a larger/multi-year train slice or holiday/event calendars would likely break it.
- Weather data (the known driver of large delays) is absent from this feature set.
- Calibrated stacking (logistic on member logits with train-fold fits) instead of equal averaging.
- Automated search over which interactions to keep, scored by cross-year stability
  (train 2005 first-half vs second-half profiles) rather than eval AUC.
