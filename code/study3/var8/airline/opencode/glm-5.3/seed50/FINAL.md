# FINAL — Airline Delay AUC Optimization

## Result

| | |
|---|---|
| **Eval AUC (100k rows, 2006)** | **0.7536** |
| **Baseline (exp #1)** | 0.7141 (+0.0395) |
| **Final commit** | `71288f0` |
| **Validation** | `./validate.sh` → **CONTRACT OK** (eval AUC via `predict_proba` with target removed: 0.7536) |
| **Budget used** | 30 experiments of 40; CPU 17,699/18,000 s; ~138 min wall |

Final `train.py` is HEAD (`71288f0`), validated at exactly this tree state.

## Final approach

XGBoost bag-of-6 with **random-subspace members** (seed bags only work here because
`colsample_bytree < 1` gives each member a different column draw; subsample 1.0 keeps all rows):

- **Features (7):** `dep_minutes` (DepTime hhmm → minutes since midnight), `sin_dep`/`cos_dep`
  (daily cycle), `dist` (Distance), and native categoricals `UniqueCarrier`/`Origin`/`Dest`
  (`enable_categorical=True`; eval levels recoded via `pd.Categorical(..., categories=train_levels)`
  so unseen 2006 categories → NaN instead of crashing predict).
- **Members:** 3 full-feature (seeds 1-3, colsample 0.85) + 3 feature-dropped
  (seed 4 drops carrier, seed 5 drops Dest, seed 6 drops dist — all colsample 1.0 to compensate).
- **Per member:** `XGBClassifier(n_estimators=450, max_depth=0, grow_policy="lossguide",
  max_leaves=512, learning_rate=0.05, min_child_weight=1, subsample=1.0, reg_lambda=1.0,
  reg_alpha=0.5, tree_method="hist", max_bin=256, enable_categorical=True, n_jobs=4)`.
- **Prediction:** arithmetic mean of the 6 members' `predict_proba` (rank-average identical).
- Train time ~72s, predict ~15s on eval — well under the 120s cap.

## What worked (kept improvements)

| Change | Eval AUC | Δ |
|---|---|---|
| Baseline (d6, depthwise, raw cats, one-hot) | 0.7141 | — |
| Drop all target/route/date encodings; dep_minutes + sin/cos + 3 native cats | 0.7138* | (*single-model regime; see below) |
| lossguide, leaves 512, mcw 1, col 0.8, bag-of-5 seeds | 0.7494 | +0.035 |
| + `dist` (kept only inside the bag), reg_alpha 0.5 | 0.7518 | +0.0024 |
| Subspace bag: 3 members drop carrier/Dest/dist | 0.7530 | +0.0012 |
| colsample 0.85 (full members) | 0.7533 | +0.0003 |
| n_estimators 450 | 0.7534 | +0.0001 |
| Per-member colsample: drops at 1.0, fulls at 0.85 | **0.7536** | +0.0002 |

Reseed checks (seeds 11-16 → 0.7528; 21-26 → 0.7532) put the seed noise floor at ~±0.0004,
so the last three rows are at/near noise — kept because each was also flat-to-positive in
single-model screens, not eval-only wins.

## What didn't work (all reverted)

- **Everything label-dependent:** target encodings (route/origin/carrier, m=25/100), route
  categorical, origin/dest/carrier×period interaction cats — 2005 year-shift makes them
  untransferable noise (internal CV said 0.785, eval said 0.71).
- **Date features:** Month, DayofMonth, DayOfWeek (even sin/cos), date strings — all hurt eval.
- **Time extras:** hour int/cat, higher harmonics, quadratic term, minute-of-hour blocks.
- **Config:** DART (0.7446), colsample_bynode (0.7517), subsample 0.9, max_bin 384/512/1024,
  max_leaves 448/576/640/1024, mcw 2/20, alpha 0.2/0.3/0.7/1.0, lambda 0/3/5, eta 0.03-0.08
  variants, rounds 300/500/600, per-member max_bin diversity, mixed depthwise/lossguide,
  month-holdout members, pair-drop member, bag 5→6→7 scaling, jitter/mixed-rate bags,
  rank-average aggregation, subsample 0.98 on fulls.

## Key insight

Train (2005) and eval/holdout (2006) are time-separated with a real distribution shift:
internal CV overestimates by ~0.035. Only features tied to the *daily operating pattern*
(time-of-day sawtooth: high late night, crash 4-5am, ramp to ~0.8 by late evening — nearly
identical across years) and airport/carrier structure transfer. Feature selection against
eval, not internal CV, was the single most important discipline; lossguide capacity
(leaves 512, mcw 1) + mild L1 and seed/subspace bagging did the rest.
