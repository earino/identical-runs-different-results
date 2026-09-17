# Final Report — Airline Delay AUC (xgboost-autoresearch, scenario 2)

**Best Eval AUC: 0.7362** (final config, commit `149909c`, exp19). Best single experiment on record:
0.7365 (exp18, a 5-granularity variant) — statistically tied with the final config (+0.0003 < 0.0005
noise bar) but ~20s slower; the faster 4-way config was kept for holdout-scoring headroom.
Baseline → final: **0.7141 → 0.7362 (+0.0221)** over 19 experiments (2 timeouts).

## Final model
12-member XGBoost ensemble = 4 departure-slot granularities × 3 seeds, probability-mean:
- Slot feature `slot_c = dep_min // {15, 10, 20, 12}` as **native categorical** (one granularity per member family); seeds 42/7/2026.
- Per member: `n_estimators=600, learning_rate=0.033, max_depth=9, min_child_weight=50, gamma=5.0, subsample=0.7, colsample_bytree=0.7, reg_lambda=1.0, tree_method="hist", enable_categorical=True, n_jobs=4` (~5.3s/fit; whole run 77s of the 120s cap).
- Features: numeric `DepTime, dep_min, dep_sin, dep_cos, Distance, log_distance`; native categorical `Month, DayofMonth, DayOfWeek, UniqueCarrier, Origin, Dest, hour_c (=hour%24), slot_c` with levels fixed from train (eval/holdout contain unseen levels, e.g. carrier `AQ` → must map to NaN); smoothed target encodings (smooth=30, unseen→prior) on `Month, DayofMonth, DayOfWeek` only.
- **Month-recency sample weights 1.0→1.5** (Dec-2005 rows closest to the 2006 eval/holdout).
- All feature engineering lives inside `prepare()`; `predict_proba(df)` rebuilds everything from a raw frame.

## Changes that mattered most (top 5)
1. **Native-categorical 15-min departure slot** (+0.0085 single config) — by far the largest win; grown into the 4-granularity slot-mix ensemble (+~0.006 from the slot family overall).
2. **Depth/regularization re-tune after the slot feature**: d5→d9 with mcw 20→50, gamma 1→5, lr .08→.033, 200→600 trees (+0.005).
3. **Seed averaging** (12-seed +0.0005 over 3-seed; noise floor ±0.002–0.003 single-seed due to colsample draws).
4. **`hour_c` native categorical** (+0.0017) after one-hot/cyclic time encodings plateaued.
5. **Month-recency weights** (+0.0004–0.0008, robust direction given the 2005→2006 shift).

## What did not help (all multi-seed verified)
1. **Anything encoding 2005-specific co-occurrences**: route/carrier/airport native categoricals, every interaction target encoding tried (carrier×hour, origin×dest, dow×slot, slot×month, carrier×slot, …), day-of-year TEs — the 2005→2006 time shift kills them all.
2. **High-cardinality airport/carrier TEs**, even with exponential time decay; TE smoothing variants (10/100 ≈ 30).
3. **Alternate learners/objectives/stacking**: DART (0.7279 vs 0.7346), lossguide, max_bin 512, mixed-depth ensembles, logit-mean and rank-mean aggregation (= prob-mean), OOF stacking (stage-B overfit stage-A's OOF feature: 0.7281 < 0.7336), early stopping on a random split (misleading best_iter), next-day flag for DepTime≥2400, q5/q20/q30 slot granularities.

## With more budget
The eval signal is exhausted (≈25 consecutive screens at noise level). Given more budget I would:
re-tune ensemble composition against a proper 2005→2006 time-split CV instead of eval.csv (5–10× more signal per decision); exploit the 120s envelope harder (vectorized/polars prep, cache reuse) to fit 20–30 members spanning depth × learning-rate × slot-granularity families; and re-tune depth/regularization jointly with the ensemble rather than sequentially, since member diversity changed the optimum's neighborhood.

*Validation: `./validate.sh` → `CONTRACT OK` (predict_proba on target-dropped frame reproduces 0.7362).*
