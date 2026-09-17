# FINAL — airline departure-delay prediction (XGBoost)

**Best Eval AUC: 0.7464** (experiment #39, commit `5e2a5f1`), up from the 0.7141 baseline.

The final `train.py` trains a 19-model XGBoost ensemble on engineered features and is
contract-valid (`./validate.sh` -> `CONTRACT OK`, eval AUC reproduced via `predict_proba`
with the target column removed).

## Changes that mattered most

1. **Composite (out-of-fold, smoothed) target encoding** of `Origin/Dest/Carrier/Route`
   crossed with `dep_hour` and `DayOfWeek`. Fitted on train only, with 5-fold OOF values
   for the training matrix so `predict_proba`/`prepare(df)` stay leakage-free. Biggest
   single lift: 0.7347 -> 0.7405 (exp #16, #17).
2. **Congestion / frequency features** — counts of scheduled flights per `origin|hour`,
   `dest|hour`, `route|hour`, `carrier|hour`, plus per-airport/route totals and their
   ratios. Lift 0.7235 -> 0.7301 (exp #9, #10).
3. **Time-of-day decomposition of `DepTime`** — wrapped hour, minute, continuous minutes,
   next-day flag, parsed numeric calendar fields (`month`, `day`, `dow`, weekend).
   0.7141 -> 0.7215 (exp #2).
4. **Strong L2/L1 regularization** (`reg_lambda=20`, `reg_alpha=4`) applied to every model.
   Monotone gains 0.7417 -> 0.7460 (exp #25-#27) and by far the most reliable direction.
5. **A 19-model lossguide + depthwise ensemble** (`max_leaves` 32-512 and `max_depth` 4-9),
   averaged in probability space, with tree counts scaled 0.7x. Diversity across grow
   policies was consistently worth more than any single tuned model (exp #14, #15, #38).

## Things that did NOT help

- **Plain categorical `Origin_Dest` route feature and simple single-key target encodings**
  (exp #3, #4, #11) — high-cardinality splits and redundant encodings were neutral or worse.
- **Adding more composite target-encoding keys** (`dest_dow`, `route_carrier`,
  `origin/carrier_month`, exp #20, #23) — sparser keys added noise and dropped ~0.002.
- **Early stopping / more raw capacity** (exp #5, #6, #30) — 2000 trees or a TS probe on a
  random 2005 holdout overfit/undershot; extra counts (`dow`, `origin_carrier`) and
  day-of-year features (exp #12, #24) were also flat.

## What I would try with more budget

Explore a two-level stack (train a logistic/linear meta-learner on out-of-fold ensemble
predictions instead of a simple average), and replace the OOF target-encoding smoothing
constant with a per-key learned shrinkage. Beyond that, the ceiling here is set by the
absence of weather and upstream-aircraft-rotation data; if those became available they
would likely dominate everything above. A careful Bayesian/random search over the
lossguide `max_leaves` x `min_child_weight` x `reg_lambda` surface would also be worth a
few dozen more experiments, since regularization was still improving at the cutoff.
