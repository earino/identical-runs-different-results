# FINAL — autoresearch XGBoost (airline delay)

**Best Eval AUC: 0.7262** (experiment #40, commit `ec19a5a`), up from the baseline 0.7141.

## Setup
XGBoost `hist` classifier, `enable_categorical=True`, 4 threads. Train on `data/train.csv`
(2005 slice), evaluate on `data/eval.csv` (2006 slice). Final model is an ensemble of 14
XGBoost members (varying depth / `grow_policy` / seed) whose `predict_proba` outputs are averaged.
All feature engineering lives inside `prepare(df)`; every encoder/statistic is fit on training
data only, so `predict_proba` reproduces the exact pipeline on the hidden holdout.

## Changes that mattered most
1. **Categorical split configuration (largest gain).** `max_cat_to_onehot=32` lets the
   low/medium-cardinality columns (Month 12, DayOfWeek 7, DayOfMonth 31, UniqueCarrier 20) be
   one-hot encoded instead of partitioned (+0.0035 combined with #2), while Origin/Dest stay
   partition-split. `max_cat_threshold=16` then tightens those partition splits, cutting
   overfitting on the 282-level airport columns (+0.0016).
2. **Regularization tuned for a time-shifted target.** Depth 4–6 with `min_child_weight=20`,
   `reg_lambda=5`, `subsample/colsample=0.8`, `learning_rate=0.03`. The unregularized baseline
   (30 trees, depth 6) and deeper/larger models (depth 7, 400 trees) both overfit 2005 and
   generalized worse.
3. **Time-of-day features.** `dep_hour`, `dep_minofday` (minutes since midnight) and `is_weekend`
   derived from `DepTime` (+0.0005).
4. **Label-free congestion / frequency encodings.** Counts of Origin, Dest, UniqueCarrier, and
   especially counts of (Origin, hour) and (Dest, hour) plus their normalized ratios
   (+0.0016). These capture stable airport-traffic structure that transfers across years.
5. **Heterogeneous ensemble.** Averaging diverse members (depth 3–7, `lossguide` with 48–127
   leaves, several seeds) added +0.0009–0.0013 and reduced variance.

Total progression: 0.7141 → 0.7262.

## What did NOT help (reverted)
- **Target encoding** of Origin/Dest/Carrier/(Origin,hour): −0.006 AUC. 2005 delay *rates* do not
  transfer to 2006 — only label-free structure does.
- **Route (Origin_Dest) as a categorical** and other high-cardinality categorical interactions:
  −0.011 AUC (severe overfit).
- **Calendar-specific counts** ((Origin, Month_Day)): −0.011 AUC, because dates don't align
  across years; likewise seasonal (Origin, Month) counts slightly negative.
- Also neutral/reverted: cyclical sin/cos features, `log(Distance)`, `max_bin=512`,
  `gamma=1.0`, stronger still regularization (λ=10/mcw=30), DART members (slow and no better),
  and simply enlarging the ensemble beyond ~14 members.

## If I had more budget
I would focus on categorical handling and validation protocol. The one-hot/threshold results show
the airport columns are the main overfitting source under the year shift, so I would sweep
`max_cat_to_onehot`, `max_cat_threshold` and per-column encoding choices (e.g. one-hot only
Origin, or rare-level pooling) more systematically. I would also build a **time-based internal
validation split** (e.g. hold out part of 2005 by month) to select `n_estimators`/regularization
without touching `eval.csv`, which would make the keep/discard decisions less noisy and reduce
the risk of overfitting the 100k eval slice. Finally, a stacked meta-learner (still XGBoost)
over the ensemble members' out-of-fold predictions might recover a little more, though the
binary AUC ceiling for this feature set looks close.
