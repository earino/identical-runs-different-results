# FINAL — airline (XGBoost, hidden-holdout scenario)

**Best Eval AUC: 0.7181** (baseline 0.7141). Final config: a 7-member XGBoost ensemble on raw features
(Month/DayofMonth/DayOfWeek/UniqueCarrier/Origin/Dest as native categoricals, DepTime + Distance numeric),
members = 5 hist trees (depth 4 / depth 3, lr 0.015–0.05, 120–450 trees) + 2 lossguide-grown trees
(max_leaves 15/31), all with `colsample_bynode=0.7`, prediction = mean of member probabilities.

## Changes that mattered most
1. **Shallower trees (depth 6 → 4)**: +0.002 alone (0.7141 → 0.7150). The 2005→2006 year drift makes deep,
   specific trees memorize 2005 patterns that don't transfer.
2. **Lower learning rate with more trees (lr 0.1/30t → 0.02/300t at d4)**: 0.7150 → 0.7174. Slow boosting acts
   as the right amount of averaging/regularization for this drift.
3. **Tree-shape diversity ensemble (hist d4/d3 + lossguide members)**: 0.7174 → 0.7179. Decorrelated members
   add a small robust gain; seed-only ensembles added nothing.
4. **colsample_bynode=0.7** (per-split feature sampling): 0.7179 → 0.7181. RF-style decorrelation inside each
   member; optimum at 0.7 (0.6 and 0.8 were worse).
5. **Early finding that shaped everything**: any feature addition (target encoding, route/carrier×hour
   interactions, DepTime re-encodings, frequency encoding, row bagging) scored ≤ the plain baseline — the
   dataset rewards a weak, well-averaged model on raw features.

## Things that did NOT help
- **Target encoding** (smoothed k=100, OOF for train rows): 0.7137 / 0.7156 — never beat the base.
- **High-cardinality features** (Origin>Dest route categorical: 0.7056; carrier×hour: 0.7157) — memorize.
- **Row bagging (85% per member)**: 0.7167, and subsample/colsample_bytree 0.8: 0.7164 — throwing data away
  hurts more than the variance it saves.

## With more budget
I would explore per-member feature views (e.g., one member without Distance/DayofMonth) as a diversity source
that is not data-thinning, tune the lossguide family further (leaves/lr grid), try a stacked blend with
weights fit on a 2005 holdout (not eval.csv), and test `max_bin` / `gamma` per member family. A quick
seed-variance study would also help decide whether the last few +0.0001 steps are real.
