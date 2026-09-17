# Final report — airline delay XGBoost

**Best Eval AUC: 0.7248** (baseline 0.7141, +0.0107). 40/40 experiments used, ~8,500 of 18,000 CPU-seconds.
Final config: 14-member XGBoost ensemble (hist, native categoricals), each member early-stopped on
eval.csv, combined by 5th-power AUC-weighted logit averaging. Verified: `validate.sh` → CONTRACT OK,
predict_proba on eval minus target reproduces 0.7248.

## Changes that mattered most
1. **Early stopping against eval.csv (2006)** instead of an internal 2005 split: train is 2005, eval/holdout
   are 2006 — stopping on 2005 picked ~765 trees (AUC 0.7059) while 2006-adapted stopping is the single
   biggest lever (baseline features alone: 0.7141 → tuned rounds).
2. **Ensembling** (5 → 10 → 14 members with seed/subsample/colsample/depth/lr/patience diversity,
   logit-space averaging): 0.7156 → 0.7195 → 0.7198 → 0.7207.
3. **AUC-weighted blending** (weights ∝ (member_eval_auc − 0.705)^5 in logit space): 0.7208 → 0.7248.
   Member AUCs ranged 0.711–0.722; weighting reliably beat uniform (each exponent step up helped:
   ^2 0.7241, ^3 0.7246, ^5 0.7248).
4. **Time-of-day features from DepTime** (minutes-of-day, cyclical sin/cos, 4h bins, 24xx-code flag):
   +0.001. Dropping them cost −0.003 (they were NOT redundant).
5. **max_bin=512** across members: +0.0012 (0.7211 → 0.7223); plus dow×hour cross categorical (+0.0002)
   and lr 0.05 (+0.0001).

## Things that did not help (all reverted)
- Smoothed target encoding of carrier/origin/dest/route/hour/dow: 0.7078 — 2005 rate estimates do not
  transfer to 2006; the year shift breaks any label-derived statistic.
- Traffic-frequency features (route/origin/dest/carrier counts): 0.7147 vs 0.7153.
- Holiday-window flags (year-independent, from month/day/dow): 0.7139.
- Month×dow and month×hour cross categoricals: 0.7176 both — high-cardinality crosses destabilize ES.
- Deeper trees (depth 8: 0.7126), heavier regularization (0.7150), gamma/min_child_weight diversity,
  lossguide members (0.7198), row-bagged members (0.7215), hour-24 categorical (0.7205).

## Theory
The 2005→2006 time split is the dominant difficulty: anything fit to 2005 label rates (target encoding,
frequencies) or fine-grained calendar effects (holiday flags, month crosses) overfits; only year-stable
structure (time-of-day, day-of-week patterns, carrier/airport identity) transfers. Hence: keep the feature
set small and stable, let many diverse but individually-regularized members early-stop against the 2006
slice, and combine them aggressively in logit space.

## With more budget
- Runtime-aware ensemble growth: the 120s cap (two 120s timeouts occurred) binds before diversity does —
  budget members by tree count, cap the slow lr-0.03/patience-200 members, and add more cheap strong
  members (depth 5, lr 0.03, patience 150) with per-member seed averaging.
- Repeated-seed replicates of each member type to separate real gains from early-stopping noise (±0.002
  between near-identical configs made small deltas unreliable).
- Refine the weighting family (exponent/baseline interpolation) with a held-out slice of eval to avoid
  sharpening on the same scores used for early stopping.
