# Final report — airline delay (XGBoost, 2005 train → 2006 eval/holdout)

**Best Eval AUC: 0.7509** (baseline: 0.7141). 17 experiments run; final `train.py` passes
`validate.sh` (`CONTRACT OK`), runs in ~80 s wall / ~3 GB peak RSS. Stopped when the remaining
Python CPU budget (~1000 s) was only enough to validate and finalize safely — further experiments
would have risked an unvalidatable final state.

## The 5 changes that mattered most

1. **"RF-ization" of the booster** (0.7141 → 0.7403): very deep trees (`max_depth=30`) + strong L1
   (`reg_alpha=3`) + heavy feature bagging (`colsample_bytree=0.3`), lr 0.02 × 700 rounds, `max_bin=512`.
   Under the 2005→2006 shift this suppresses year-specific noise that plain shallow boosting memorizes.
2. **Robust interaction codes as native categoricals** (→ 0.7468): `carrier×hour`, `origin×hour`,
   `dest×hour`, `carrier×distance-bin`. Congestion/schedule patterns of carriers and airports by time of
   day repeat across years, unlike route-level or date-level effects.
3. **`max_cat_to_onehot=100`** (→ 0.7500): per-level one-hot style categorical splits instead of
   partition-based splits — the single biggest hyperparameter win of the late phase.
4. **Coarse 2-hour origin/dest codes + `min_child_weight=3`** (→ 0.7509): complementary granularity
   for one-hot splits; slightly larger leaves stabilize them.
5. **Time features from DepTime with post-midnight wrap** (needed by everything else): `hour`,
   `minute`, `dep_h = hour + minute/60`, with `DepTime > 2400` (i.e. 24xx/25xx = past-midnight) wrapped
   via `% 24` so 1:05am (25:05) aligns with 1:05am.

## 3 things that did not help (all reverted)

1. **Route/airport target encodings and route codes** — any Origin×Dest-derived feature was poison
   under the year shift (0.7141 → ~0.700–0.706). 2005-specific route delays do not transfer.
2. **More boosting rounds at high learning rate + internal early stopping** — a 2005 validation split
   picks *more* rounds than the 2006 distribution supports; eval AUC fell monotonically with rounds
   at lr 0.1. Fixed moderate rounds + low lr was much better.
3. **Ensembles (seed/subsample/two-view), DART, lossguide, interaction/monotone constraints at depth,
   calendar-date codes (Month×DayofMonth), hour×month/dow codes, congestion counts, sample weighting** —
   all flat or negative; several (monotone constraint, max_bin=512) helped at shallow depth but hurt
   after the deep-tree/one-hot regime change.

## What I would try with more budget

The eval plateau near 0.751 looks drift-limited, not capacity-limited, so I would attack the
distribution shift directly: (a) time-blocked validation *within* 2005 (last months as pseudo-2006) to
make hyperparameter choices without touching eval, giving unbiased estimates of what transfers;
(b) importance weighting / density-ratio correction on drift-stable covariates (month, hour, carrier)
to reweight 2005 toward the 2006 marginal; (c) a systematic "stability screen": per-feature
contribution compared across 2005 month-blocks, dropping features whose contribution does not replicate;
(d) a small rank-blended ensemble of the current config with a coarser-bin variant (2h/3h codes) and a
differently-seeded one-hot model — with only ~600 CPU-s per model this must be budgeted carefully;
(e) finer sweeps of `max_cat_to_onehot`, `min_child_weight` and code granularity under the one-hot
regime, which was only sampled coarsely here.
