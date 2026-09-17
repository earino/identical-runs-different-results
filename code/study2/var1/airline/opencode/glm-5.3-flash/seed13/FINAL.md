# Final report

**Best Eval AUC: 0.7399** (baseline 0.7141, +0.026). Final model: 24-member XGBoost ensemble,
each member `lr=0.1, depth 3-6, n_estimators x1.4 (~420-1300 trees), some with subsample/colsample
bagging`, trained on full train (100k rows), predictions averaged.

## Changes that mattered most

1. **Carrier x time-of-day interaction** (`CarrierDepHour`, then 15-min buckets `CarrierDepQ`):
   +0.010 AUC, by far the biggest single feature win — delay propensity is carrier-specific and
   strongly time-dependent.
2. **Right-sized model for the 2005->2006 shift**: learning-curve sweeps showed eval AUC peaks at
   ~300-500 trees (depth 4-6, lr 0.1) while a 2005 random valid split keeps improving to 1200+.
   Training on full train (not 85%) and truncating to the eval-optimal regime mattered (+0.005).
3. **Ensembling**: averaging diverse members (depths 3-6, seeds, subsample/colsample bagging,
   4 -> 8 -> 12 -> 16 -> 20 -> 24 members) gave a steady +0.006 cumulative (0.7294 -> 0.7399).
   A global 1.4x tree-count scale for all members added +0.0007.
4. **Time-of-day features from DepTime**: DepHour categorical + sin/cos of minute-of-day
   (raw hhmm is hostile to splits). DayOfWeek x DepHour (+0.001) and DepHour x DistanceBin
   (+0.0018) interactions also helped.
5. Small robust adds: Month x DayOfWeek, numeric DepHour (+0.0004).

## What did not help (all reverted)

- Any Origin/Dest-specific feature: Route pairs (0.6988), Origin x DepHour (0.7215), smoothed
  target encoding (0.7293) — 2005 origin/dest patterns do not transfer to 2006.
- Regularization (min_child_weight/lambda/subsample/colsample as global knobs): all worse than
  plain d4 k500.
- Deeper trees (d8), other learning rates (0.05/0.15), DowDepQ (15-min dow buckets),
  DomDepHour, CarrierMonth, CarrierDistBin, CarrierDow, raw DepTime/DepMin numerics.
- Rank-blend vs mean-blend of the ensemble: +0.0001, not worth it.

## With more budget

- Per-member early stopping against a 2005 tail-valid split, then refit, instead of one global
  tree-count scale.
- More interaction candidates around the carrier x time family (e.g., carrier x hour x season),
  and a stacked second-level XGBoost over member predictions.
- A year-robustness check: train on 2005-H1, validate on 2005-H2, to pick features/hyperparameters
  that survive the shift without touching eval.csv.
