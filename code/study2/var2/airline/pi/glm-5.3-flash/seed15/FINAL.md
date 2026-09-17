# Final Report — Airline Delay Prediction (AUC maximization)

**Final eval AUC: 0.7498** (40/40 experiments; baseline 0.7141 → **+0.0357**)
**Final commit:** `bb360e6` ("simplify: drop Dlog/wrap/is45")
**Validation:** `./validate.sh` → CONTRACT OK; `predict_proba` reproduces 0.7498 on eval with the target
column removed. Run-to-run seed variance measured ≤ 0.0002 (SEED 42 vs 999 identical scores).

## Final architecture (train.py)

### Feature engineering — all inside `prepare(df)`, fitted on train only
- Raw categoricals (train-fitted `pd.Categorical` levels): **Month, DayOfWeek, UniqueCarrier, Origin, Dest**
  (DayofMonth raw cat deliberately **dropped** — 31 noisy levels; removing it gained +0.003)
- Distance (numeric; log1p variant tested redundant and dropped)
- DepTime → `dep_minutes`, `dep_hour`, sin/cos of minutes-mod-1440 (red-eye 24:00–26:20 wrap handled cyclically)
- `dep_mm` + flags `is00`, `is30` (scheduled-block artifacts; is45 dropped as redundant)
- Ordinal ints Month_i/Dom_i/Dow_i, month sin/cos, `is_weekend`
- `teH`: smoothed hour-rate target encoding (k=10); `dep_hour_cat` (27 levels)
- **`car_h`**: carrier×hour categorical (540 levels) — single biggest win (+0.011)
- `dh_cat`: distance-quintile×hour categorical (fixed train quantile edges; essential, −0.004 if dropped)
- `teDH` dow×hour TE (k=30), `teMH` month×hour TE (k=30), `dom_b` (4 day-of-month bins)

### Model
- **5-fold bag**: 5 folds × depths {4,5,6,7,8} → XGBClassifier per (fold, depth) on the other 80%
- **5 full-train members** (one per depth) → 30 models total
- Params: `n_estimators=400, lr=0.1, tree_method="hist", enable_categorical=True`
- `predict_proba(df)` = mean of the 30 members' positive-class probabilities
- Runtime ≈ 90 s wall (safe margin under the 120 s cap; n=500 flirted with it and was reverted)

## Score trajectory (40 experiments, all logged in experiments.tsv)
| # | change | eval AUC |
|---|--------|----------|
| 1 | baseline (categorical XGB d6) | 0.7141 |
| 5 | FE + teH + hour-cat + car_h | 0.7293 |
| 6 | 5-fold bag | 0.7329 |
| 8 | minute flags | 0.7393 |
| 9 | dh_cat, cd_cat | 0.7438 |
| 11 | 1 seed/fold, depths 4–8 (25 models) | 0.7450 |
| 17 | + 5 full-train members (30 models) | 0.7452 |
| 18 | drop DayofMonth categorical | 0.7486 |
| 24 | + teDH/teMH/dom_b, n400 | 0.7492 |
| 34 | ablation: drop cd_cat (became redundant) | **0.7498** |
| 36 | simplify: drop Dlog/wrap/is45 (equal AUC) | **0.7498** |

## What mattered most (top 5)
1. **High-cardinality interaction categoricals** — carrier×hour, distance×hour (fold bagging + these ≈ +0.02).
2. **5-fold × depth-diverse bagging (d4–d8) + full-train members** (+0.014 over single d6).
3. **Scheduled-time artifacts**: exact-minute flags is00/is30, hour categorical, smoothed hour TE (+0.01).
4. **Pruning noise**: dropping the raw DayofMonth categorical (+0.003) and later cd_cat (+0.0006).
5. **Keeping raw Origin/Dest/Carrier categoricals** — dropping them collapses AUC to ~0.69.

## What did not help
- Target encodings replacing cats for Origin/Dest (2005→2006 rate shifts up to ±0.12 break them);
  day-of-month interactions in any form (−0.02 to −0.04); route (Origin×Dest) categories.
- Single-model regularization (subsample, colsample, gamma, min_child_weight, early stopping) —
  all gains vanished or inverted at bag level; the bag is already regularized. Bootstrap members ≈ folds.
- Specialist models per time-of-day segment, stacked meta-learner (overfit OOF), seeds-vs-depths
  (depth diversity won), lossguide members, max_bin=512, alpha/lambda regularization.

## With more budget
I would (a) tune the car_h/hour-cat feature block at bag level (it dominates feature importance),
(b) try ordinal target encoding of Origin/Dest with year-robust within-airport ranks, and (c) push
bag size (2 fold partitions × depth grid) if the runtime cap allowed.

## Caveat on the reported number
~25 keep/drop decisions were made against the same 100k eval sample (SE≈0.002 each), so the final
selection carries a small optimistic bias (~+0.003 max-statistic). The hidden 2006-slice2 holdout is
expected to land slightly below 0.7498; large-effect choices (features, bagging) dominate the result,
micro-selections (<0.0005 each) contribute marginally.
