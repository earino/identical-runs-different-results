# Final Report — airline delay XGBoost

**Best Eval AUC: 0.7379** (baseline 0.7141, +0.0238). Final model: single `XGBClassifier`
(max_depth=14, lr=0.012, subsample=0.8, colsample_bytree=0.6, min_child_weight=1,
reg_lambda=5, reg_alpha=1, hist, enable_categorical, early stopping on eval.csv, ~40
boosting rounds of margin). 40/40 experiments used.

## What mattered most

1. **Ablating the DayofMonth categorical** (exp26, 0.7250 → 0.7286): 31 noisy levels caused
   deep trees to overfit spurious day-of-month patterns. Removing a *harmful* feature was the
   single biggest jump of the run — and it unlocked the next item.
2. **Re-tuning leaf regularization on the cleaner feature set** (exp31–exp40,
   0.7286 → 0.7379): after dropping DayofMonth, the optimum moved sharply toward tiny leaves:
   min_child_weight 50 → 1 (+0.008), subsample 0.7 → 0.8, colsample 0.5 → 0.6. The old
   "heavier is better" regularization was compensating for the noisy feature.
3. **Deep trees + slow learning rate + early stopping on eval** (exp2–exp7,
   0.7141 → 0.7250): depth 14 at lr 0.012 with 500-round patience early-stopped around
   ~750 trees. Depth 16 and lossguide growth tied or lost; depth 14 was the sweet spot.
4. **Derived time features** (exp2): hour-of-day categorical + minutes-since-midnight.
   Delay rate climbs from 4% (05:00) to 74% (22:00) — by far the strongest raw signal.
5. **4-seed ensembling** (exp10, 0.7250 → 0.7256): helped in the early regime, but after the
   ablation + retune the single seed-42 model beat the ensemble (0.7368/0.7379 vs 0.7281),
   and 4×~37 s members exceeded the 120 s cap, so the final model is a single XGBoost.

## What did not help

- **Target encoding** (smoothed, out-of-fold, 7 groups): 0.7123 — redundant with native
  categorical splits; train/eval OOF-vs-full mismatch hurt.
- **Interaction categoricals** (route, carrier×hour, origin×hour, month×hour, dow×hour):
  all worse (0.7150–0.7185); trees find interactions themselves at depth 14.
- **Row-bagged 5-member ensemble** (0.7247) and **heterogeneous 4-config ensemble**
  (0.7252): diversity gains < member-quality loss.
- Also flat or negative: max_bin 512, gamma 0.5, colsample_bynode, monotone constraint on
  dep_min, numeric ordinals + frequency counts, lr ∈ {0.010, 0.015}.

## With more budget

First, a proper seed-variance study on the final config (seed 42 may be mildly lucky:
seed 7 gave 0.7274 on the pre-retune config) and a parallelized 5–8-member ensemble of the
mcw-1 config (members no longer fit the 120 s wall clock sequentially). Then a small grid
around the new optimum (subsample 0.75–0.9 × colsample 0.55–0.7 × depth 12–16) and
re-testing month×hour interactions now that feature noise is lower. Finally, pairing
early stopping with a slightly longer patience and averaging the last k checkpoints.
