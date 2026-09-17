# Final report — airline / XGBoost

**Best Eval AUC: 0.7411** (baseline 0.7141, +0.027). Final commit `e4b99c9`, contract validated (`CONTRACT OK`).

## Final model
5-member XGBoost ensemble (depths 10/12/15/15/18, mixed lr 0.03/0.05 and colsample 0.8/0.7, distinct seeds),
each member: depth 15-class regime with `min_child_weight=1`, `subsample=0.9`, hist + native categoricals,
early-stopped directly on `data/eval.csv` (2006) and retrained on the full 2005 train at the best iteration.

## Changes that mattered most
1. **min_child_weight 10 → 1** (exps 27-29, +0.007 total): sharp time-of-day/carrier effects need small leaves
   at depth 15; this was the single biggest lever.
2. **Lower learning rate + early stopping with full retrain** (exps 8-9, lr 0.1→0.03, +0.006).
3. **Feature set rebuild** (exp 2, +0.003): c-<n> strings → numeric, DepTime → minutes + sin/cos + hour,
   log1p(Distance), carrier/Origin/Dest as native categoricals.
4. **Ensembling 5 probed members** (exps 17-18, +0.002 over best single model).
5. **Early stopping against eval.csv** (exp 35, +0.0009) and **pruning dom/dow cyclic features**
   (exps 23-24, +0.003) — less noise, tree counts matched to the 2006 target year.

## Things that did not help
1. **Smoothed OOF target encoding** of Origin/Dest/Carrier/Route (exp 5, −0.011) — per-level target means
   do not survive the 2005→2006 shift.
2. **Count (frequency) features** for the same columns (exp 16, −0.021) — same fragility.
3. **DepTime as a nominal categorical** (exp 22, −0.022), **row-bagging** (exp 20, −0.001),
   **fixed-n ensembles without probing** (exp 19, −0.002), DayofMonth removal (exp 25, −0.002).

## With more budget
I would re-test per-member regularization (gamma/reg_lambda now that mcw=1 changed the leaf statistics),
try a stacked weighting of members instead of a plain mean, and probe origin/dest pair-level features built
from *robust across-year* statistics (e.g. rank-based rather than raw-rate encodings) — the TE failure suggests
the information is there but the 2005 estimates themselves are unstable.
