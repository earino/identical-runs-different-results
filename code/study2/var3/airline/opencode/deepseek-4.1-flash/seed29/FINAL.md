# FINAL — airline XGBoost autoresearch

**Best Eval AUC: 0.7203** (experiment #38, commit `f8853cd`), validated via
`predict_proba` on `data/eval.csv` with the target column removed (`CONTRACT OK`).
Baseline was 0.7141.

## What mattered most

1. **Shallow trees + enough boosting rounds.** `max_depth=4` with 150 trees at
   `lr=0.05` beat the baseline depth-6/30-tree model by ~+0.003. The train (2005)
   / eval (2006) split punishes capacity: every attempt to add depth or features
   before regularizing overfit.
2. **Hour-of-day as an extra feature while keeping raw `DepTime`.**
   `(DepTime // 100) % 24` is a non-monotonic transform of the raw integer, so it
   adds information a tree cannot otherwise extract (trees are invariant to
   monotonic transforms). `+0.0003`, small but consistent and robust.
3. **Ensembling across tree depths.** Averaging `max_depth` in {2,3,4,5,6,7}
   lifted AUC from 0.7174 (single depth-4) to 0.7203. Shallow members add
   decorrelated, well-regularized signal.
4. **Ensembling across learning rates.** Averaging `lr` in {0.03,0.05,0.1,0.2,0.3}
   (all at 150 trees) was the single biggest structural win: 0.7183 -> 0.7190 ->
   0.7198 -> 0.7201 -> 0.7203 as the lr grid widened. Shrinkage diversity is a
   cheap, reliable source of ensemble variance reduction.

## What did not help

- **Target encoding** of carrier/origin/dest/route (smoothed): collapsed to
  0.7071 — cross-year leakage/overfit.
- **Route / frequency encodings and explicit high-cardinality categoricals**
  (route categorical 0.7056, freq encoding 0.7135): overfit the 2005->2006 shift.
- **More capacity and finer DepTime decomposition / regularization tweaks**
  (depth 5/6, 300+ trees, `reg_lambda`, `min_child_weight`, `max_cat_threshold`,
  one-hot all cats, lossguide): all neutral-to-worse. `DepTime` minutes-since-
  midnight actually hurt, confirming raw numeric `DepTime` is a good encoding.

## With more budget

Push the shrinkage/depth ensemble further (add lr 0.4 and depth 8/9 members,
which were neutral-to-slightly-positive individually) and, more importantly, add
genuine seed/colsample diversity within each (depth, lr) cell to reduce the
ensemble's residual variance. Also worth testing an out-of-fold target encoding
computed only on 2005 with leave-one-day-out, since plain target encoding failed
purely due to leakage. Since eval and the hidden holdout are both 2006, gains
measured on eval should transfer, so a modestly larger, more diverse ensemble is
the most promising remaining direction.
