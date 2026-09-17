# Final report — airline dep_delayed_15min, XGBoost

Best Eval AUC: **0.7347** (baseline: 0.7141, +0.0206)

Final model: bag of 60 bootstrap-replicate XGBoost classifiers (5 depth/colsample configs x 12
replicates), 110 trees each, lr 0.1, reg_lambda 60, gamma 1.0, native categorical encoding,
trained on data/train.csv (2005), evaluated on data/eval.csv (2006).

## The 5 changes that mattered most

1. Hour interactions (+0.003 then compounding): Origin x hour, Dest x hour, Carrier x hour as
   categorical interactions. Evening delay accumulation is airport- and carrier-specific; this was
   the single largest feature gain (0.7173 -> 0.7201 -> 0.7209).
2. Heavy L2 regularization (+0.007 over the sweep): reg_lambda 2 -> 5 -> 8 -> 15 -> 30 -> 60 rose
   monotonically under 2005->2006 temporal drift (0.7284 -> 0.7311 -> 0.7324 -> 0.7338 -> 0.7344 ->
   0.7346). Withheld-year drift punishes sharp leaves far more than in-sample fit.
3. Bootstrap bagging (+0.001 as a lone change, more later): averaging 10-150 decorrelated members
   smoothed predictions; depth (4-8) and colsample (0.7-1.0) diversity added small consistent gains.
4. More trees per member once features were right (+0.0027): 30 -> 50 -> 60 -> 70 -> 90 -> 110 trees
   per member only paid AFTER the hour interactions existed; before that, >30 trees overfit 2005.
5. Derived time features (hour, minute, minutes-since-midnight): hurt at single-model capacity
   (0.7127) but helped inside the ensemble (part of the 0.7165->0.7173 stack).

## 3 things that did not help

1. Target/interval encodings of 2005 statistics: route delay-rate target encoding (0.7190) and
   month x origin interactions (0.7147) — year-specific rates do not transfer.
2. Day-of-week interactions (0.7196) and a raw 4198-level route categorical (0.7056) — noise and
   split-budget waste.
3. Capacity moves in the wrong regime: single 2000-tree model with early stopping (0.7090),
   depth-3 shallow bag (0.7163), lr 0.05 members (flat, slower). min_child_weight=5 was flat.

## What I would try with more budget

The lambda curve was still rising at 60 and gamma only got one probe; a joint
(lambda, gamma, min_child_weight, trees) coordinate sweep around the current optimum is the
cheapest next win. Second, an out-of-fold stacking layer: use the 60-member bag to generate
cross-fitted features for a second-stage XGB — under heavy regularization the ensemble's own
disagreement (variance across members per row) is a feature I never exploited. Third, replacing
the raw route categorical with count-based hierarchy (origin/dest traffic quantiles, carrier-route
volume) rather than delay-rate encodings, since volume statistics transferred (flat but not
harmful) while delay-rate statistics did not. Finally, an internal 2005 time-based validation split
(months 10-12 as pseudo-2006) would let me select hyperparameters against drift instead of
against eval.csv, which should make small keep/discard calls more robust to hidden-holdout shift.
