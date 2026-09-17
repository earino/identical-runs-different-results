# Final Report

**Best Eval AUC: 0.7258** (baseline: 0.7141, +0.0117)
Final `train.py` = commit `d7ee983` ("ensemble: 20 members, 4 lossguide"). `./validate.sh` → `CONTRACT OK`.

## What mattered most

1. **Heterogeneous XGBoost ensemble** (biggest lever, 0.7141 → 0.7258): 16 depthwise members spread over
   max_depth 5–10, colsample_bytree 0.4–0.9, subsample 0.7–0.9, plus 4 lossguide (leaf-wise) members with
   max_leaves 31–256, predictions averaged. Diversity across members beat every single-model tuning.
2. **Engineered numeric features** (0.7080 → 0.7149 at equal capacity): cyclic sin/cos of DepTime (3
   frequencies), hour, 4-way time band, numeric month/dow/dom, dom-edge flag, dist + log-dist — fed to
   trees alongside XGBoost's native categorical splits. Raw `DepTime` hhmm alone is nearly orderless to a
   tree (hundreds as the high-order digit); the cyclic/band features fix that and were worth +0.0069.
3. **max_bin=1024** (0.7186): finer bins let splits carve the fine-grained scheduled-departure time.
4. **Model capacity**: 400 trees × lr 0.05 × subsample 0.8 × colsample_bytree 0.8 at depth 8 was the sweet
   spot for the single model; d10/600 trees and stronger regularization (min_child_weight, gamma,
   colsample_bynode) all regressed.
5. **20 members instead of 16** (0.7256 → 0.7258): +0.0002 — real but marginal; kept because eval was not
   worse, and averaging more members should generalize slightly better to the hidden holdout.

## What did not help (all reverted)

- **Target/frequency encodings** (exp 2–3, 0.7025–0.7121): shrinkage-encoded origin/dest/route/carrier
  delay rates and frequency bins actively hurt — the 2005→2006 shift flips group delay-rate ranks
  (e.g. January 0.54→0.44), so train-fit group statistics mislead.
- **High-cardinality composite categoricals** (exp 13, 27: 0.7027–0.7030): route (≈6.6k levels) and
  DayOfWeek×DayofMonth (217 levels) as native categoricals overfit badly; even a 48-level month×band
  numeric composite (exp 38) regressed. Only low-cardinality native columns and numeric features survive
  the year shift.
- **Data/regularization tweaks around the ensemble** (exp 20, 23, 24, 28–39): min_child_weight=5,
  per-member row bagging (80%), early stopping on a 25% val split (loses 25% training data), rank-average
  combining, per-member feature-block subsets, gamma diversity, max_bin 2048, 450-tree members — all
  equal or worse. The 16-member depthwise+lossguide trade (exp 36) also lost.

## With more budget

I would (1) push the ensemble toward ~30 members with a wider lossguide/depth grid, trimming members by
their held-out contribution (needs out-of-fold predictions — budget was the blocker); (2) blend a
deeper-stalked config family (pairs of shallow+deep members) since depth diversity drove most of the gain;
(3) test monotone constraints on the cyclic time features; (4) try per-month probability calibration
fitted on 2005 and shrunk hard toward the global rate, to directly attack the year drift that killed
every group-statistic idea.
