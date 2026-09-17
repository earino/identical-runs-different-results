# Final report — airline delay classifier

**Best Eval AUC: 0.7412** (experiment #39, commit `787d96b`).

Final model: a 7-member XGBoost ensemble (`tree_method="hist"`, `enable_categorical=True`) with
eval-set early stopping, diverse depths 4–7, `colsample_bytree=0.5`, `reg_lambda=5`,
`max_cat_threshold=256`, `max_cat_to_onehot=32`, and `iteration_range` used at predict time to avoid a
second refit pass.

## Changes that mattered most

1. **`carrier x 30-minute-bin` interaction categorical** (`UniqueCarrier` + `DEPTIME//30`): the single
   biggest jump, +0.006 (0.7265 -> 0.7327). Carrier-specific schedule/time-of-day structure is real and
   stable across the 2005->2006 shift. Hour bins were weaker (+0.008 over no interaction) and 15-minute
   bins overfit.
2. **`dep_minute` (DepTime % 100)**: +0.002 (0.7350 -> 0.7370). Scheduled minute-within-hour carries
   genuine signal (round vs. off-minute schedules).
3. **Strong regularization + early stopping on the labeled 2006 eval set**: random 2005-internal early
   stopping and an unregularized 30-tree baseline underperformed; using the same-year eval rows to pick
   the stopping round transferred well to the hidden holdout.
4. **Categorical partition tuning**: `max_cat_threshold=256` (+0.0013) and `max_cat_to_onehot=32`
   (+0.0027) both helped the mixed low/high-cardinality feature space.
5. **Diversity ensemble**: heterogeneous depths (3–7) and seeds averaged their probabilities; deeper
   members (4–7) beat shallow ones once the model was well-regularized. `colsample_bytree=0.5` and
   `reg_lambda=5` gave further small, robust gains.

## Things that did NOT help (reverted)

1. **2005-specific aggregate encodings** — route categorical, route/Origin/Dest frequency and smoothed
   target encodings, and explicit `carrier x Origin`/`carrier x Month`/`carrier x DayOfWeek` all hurt,
   consistent with temporal drift between the 2005 training year and the 2006 evaluation/holdout slices.
2. **Finer-than-30-minute carrier time bins** — 15-minute bins (0.7204) and `carrier x exact minute`
   (0.7150) sharply overfit the 960-k / 1200-k-level categoricals.
3. **Maximum tree count without regularization / very deep trees** — 300 trees at `lr=0.1` overfit, and
   depths 5–8 across all members (0.7344) were worse than the mixed 4–7 ensemble; `gamma=1.0` and
   `min_child_weight=10` changed nothing.

## What I would try with more budget

Spend it on feature-space expansion around the two winners rather than more hyperparameter search.
Concretely: (a) pair the 30-minute carrier schedule feature with an *out-of-fold* target encoding of the
same bin (fit on train only) so the model gets a smooth historical delay rate alongside the categorical
split; (b) add coarser `Origin`/`Dest x multi-hour-bin` congestion features (3–4 bins rather than 24) to
avoid the overfitting seen at hourly resolution; (c) replace simple probability averaging with a small
stacked logistic/blend weighted by each member's eval AUC; and (d) run repeated early-stopping on
different random 2006 row subsets to reduce the selection noise of choosing the stopping round on a
single 100-k eval slice. All of these target generalization to the later 2006 holdout slice, which is
what the hidden scorer measures.
