# Final report — airline delay AUC (XGBoost)

**Best Eval AUC: 0.7476** (baseline 0.7141, +0.0335). Final model: ensemble of 8 XGBoost
classifiers averaged in probability space; every member trains on all 100k rows of
`data/train.csv` with early stopping (patience 25) against `data/eval.csv`.

Final config: pairs of members at learning_rate {0.01, 0.02, 0.03, 0.05} (seeds 42/7, 123/2024,
555/999, 31/77), max_depth 16, subsample 0.7, colsample_bytree 0.6, `tree_method="hist"`,
`enable_categorical=True` for UniqueCarrier/Origin/Dest. Features: Month/DayofMonth/DayOfWeek
parsed from `c-<n>` to ints, raw DepTime + dep_hour + dep_min, Distance + log1p(Distance).

## Changes that mattered most

1. **Train on all 100k rows with early stopping on eval.csv** (exp 3, 0.7254 vs 0.7172): more data
   plus an ES signal drawn from the same 2006 distribution as the hidden holdout.
2. **Deep trees**: max_depth 8 → 16 (exp 6–8, 0.7254 → 0.7381). Depth 20 plateaued; depth 16 kept.
3. **Low learning rate**: 0.05 → 0.01 (exp 15, 0.7399 → 0.7456 at depth 16 with sub 0.7/col 0.6).
4. **Stochastic regularization**: subsample 0.7 / colsample_bytree 0.6 (exp 10, +0.0011 over 0.8/0.8).
5. **lr-diverse 8-member ensemble with prob averaging** (exp 34–36, 0.7456 single → 0.7476): adding
   member pairs at lr 0.02/0.03/0.05 decorrelates the average; each +pair gave +0.0002–0.0006.

## Things that did not help

- **Target / frequency encodings** (exp 4, 5, 24): train-fitted statistics from 2005 mislead on 2006
  (year shift); they hurt on both the weak and the strong base model.
- **Feature bloat**: route categorical (~11k levels), month×dow / hour×dow / carrier×hour crosses,
  cyclic sin/cos encodings (exp 4, 18*, 25–27) all cost 0.01–0.02 AUC — the model wants a lean set.
  (*exp 18 ran on a wrong base config, but crosses were retested cleanly and still hurt.)
- **Micro-tuning dead ends**: min_child_weight 10 (0.7387), gamma 1.0 (0.7268), depth 20 at lr 0.01
  (0.7278), max_bin 512 (+0.0001), rank- vs prob-averaging (identical), ES on eval_metric="auc"
  (0.7277 — the sklearn API treats AUC as minimize, stopping at an early dip), colsample_bynode
  and 5+ member ensembles (timeout at the 120 s experiment cap).
- Pruning "redundant" DepTime features (raw deptime/dep_min/log_distance) collapsed AUC to 0.7199 —
  minute-level DepTime granularity is one of the strongest signals in the set.

## With more budget

- Larger ensembles via 2-thread parallel fits (wall-time halves per member) and member-level
  row-bagging for extra diversity.
- Careful re-test of prediction-time-robust aggregates (e.g., iteration-range snapshot ensembles)
  and a calmer lr sweep (0.008–0.015) with depth 14/16 per member.
- A hedge against eval-overfit: average ES-on-eval members with members early-stopped on the last
  20k rows of train (time-split proxy), which may transfer better to the hidden holdout.
