# Final report — autoresearch XGBoost (airline, scenario 2)

Best Eval AUC: **0.7544** (baseline 0.7141, +0.0403). Final `train.py` = HEAD (exp18),
validates with `CONTRACT OK` (`validate.log`: predict_proba path reproduces 0.7544).

## What the final model is

Equal-weight average of 22 XGBoost members over 4 feature views, all with native categorical
splitting (`enable_categorical=True`, category levels fixed from training data only):

- 16 classifiers: 8 tuned (n1600/lr0.04, d4-d6, subsample .8, colsample .5-.7, mcw 10-20,
  lambda 5, alpha 1) over views `full` (carrier x half-hour interaction), `full_qh`
  (carrier x quarter-hour), `geo` (slot + Origin/Dest + Distance), `minimal`
  (DepTime + Distance only), plus seed bags; 8 "bare" members (n800/lr0.08, xgboost default
  regularization) that add inductive-bias diversity (+~0.002).
- 6 squared-error regressors on 0/1 labels (`reg:squarederror` via `xgb.train`, outputs
  clipped to [0,1]) — different loss shape, decorrelated errors (+~0.001).
- 13 of 22 members train with month-recency sample weights (1 + 0.15*(month-1); Dec 2005
  weighs 2.65x Jan) — the eval/holdout year (2006) is ahead of every training month.

## Changes that mattered most

1. **DepTime as scheduled-time-of-day features** (exp2/5): hour/min numeric, cyclical
   sin/cos, hour-slot categorical. Delay rate is a steep monotone function of departure
   time (4% at 5am -> 74% by 10pm); the baseline only saw raw `hhmm` values.
2. **Carrier x half-hour categorical** (exp7, +0.016 alone): a native categorical
   interaction (480 levels) that lets single splits separate e.g. "WN at 6:30pm" from
   "WN at 10am". Finer (quarter-hour) worked in one member family; finer still (exact
   minutes) overfits and collapsed.
3. **Low capacity + strong regularization** (exp6, +0.007): depth 6 -> 4, colsample 0.8 ->
   0.5, min_child_weight 5 -> 20, lambda 5, alpha 1. The 2005->2006 shift punishes deep
   trees; d6+n1000 dropped to 0.7088 while d4 regularized gained.
4. **View-diverse ensemble** (exp8-13, +0.021 total): members over different feature views
   (full/full_qh/geo/minimal) and different regularization regimes average to +0.01 over
   any single member; seed bags and row-bagged (subsample .5) members add the rest.
   Diversity of inductive bias mattered far more than more tuned seeds of one config.
5. **Month-recency weighting** (exp10, +0.0013) and **loss diversity** (exp14-17,
   +0.0015): small but consistent gains aimed directly at the temporal drift.

## What did not help

- Target/count encoding of Origin/Dest/route (OOF or not): 0.7136-0.7159 — native
  categorical partitions already capture it.
- Route, carrier x origin, carrier x dest, dow x dom, holidays/doy features, day-of-month
  cyclicals: every one flat to negative (high-cardinality 2005-specific combos).
- Stacking / OOF-learned member weights / meta-learner: OOF AUC 0.80 but eval 0.742 (<
  simple mean) — weights latch onto 2005-specific member rankings that don't transfer.
- rank:pairwise / rank:ndcg objectives (0.500 — degenerate with one global group),
  interaction constraints (-0.026), monotone constraints, exact-minute carrier cats
  (0.7015), max_bin/lossguide, time-forward early stopping (confirmed n~1600 already right).

## With more budget

- Larger member count of the two winning families (bare-default and squared-error
  regressors) across more feature-view combos, at ~2x ensemble size (the eval-side gains
  had not fully flattened: each new diverse family still added +0.0005-0.001).
- A second recency axis: per-carrier drift (some carriers' delay rates shift more than
  others) as a dynamic reweighting rather than a global month weight.
- grown-then-pruned ("elastic") ensembles: train more candidates cheaply and select the
  subset by a *pre-registered* rule to avoid the eval-shopping trap this run mostly avoided.
