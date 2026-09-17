# FINAL — airline dep-delay XGBoost (autoresearch benchmark)

**Best Eval AUC: 0.7592** (experiment #6, commit `54cd430`), up from the 0.7141 baseline
(+0.045). `./validate.sh` prints `CONTRACT OK`; `predict_proba` reproduces 0.7592 on eval.csv.

## Final recipe (train.py)

- Features (all built inside `prepare()`, stats fitted on train.csv only): numeric month/dom/dow,
  raw DepTime, hour, minute, minutes-of-day + sin/cos, distance + log-distance, native categoricals
  (carrier/origin/dest), and 9 label-free count/volume features (flights per origin, per route,
  per origin×hour, per carrier×hour, per route×hour, plus 2h/3h block variants).
- Model: mean ensemble of 5 XGBClassifier seeds, each `max_depth=18, learning_rate=0.05,
  n_estimators=150, min_child_weight=1, subsample=0.7, colsample_bytree=0.35, reg_lambda=5`,
  tree_method="hist", trained on 100% of train.csv with fixed rounds (no early stopping).

## The 5 changes that mattered most

1. **Replacing internal-validation early stopping with full-data + fixed rounds.** The internal
   2005 split is actively harmful: val-AUC keeps rising (to 0.751) while 2006 eval-AUC peaks around
   round 50-100 and then *declines* — a big 2005→2006 distribution shift. Early stopping on 2005
   data selects rounds that reward year-specific overfit.
2. **Deep trees + strong column/seed decorrelation** (d18, colsample_bytree 0.35, 5-seed averaged
   ensemble): each deep tree overfits 2005 noise, but the ensemble averages it away while keeping
   the stable signal. Depth 6→18 with matching colsample was worth roughly +0.02 AUC on its own.
3. **Numeric time features + sin/cos cyclical encoding** of minutes-of-day (raw hhmm integers have
   an artificial 559→600 cliff; hour-of-day is the dominant predictor of evening delay buildup).
4. **Label-free count/volume features** (origin/route/carrier × hour traffic): they transfer across
   years, unlike label-based encodings. Best single batch of additions after the time features
   (the route×hour and block-hour counts added ~+0.002).
5. **Fixing a real bug found by cross-checking offline vs official results**: the count maps stored
   raw keys instead of `value_counts()`, silently zeroing all count features (0.7464 → 0.7567).

## 3 things that did NOT help

1. **Target encoding of Origin/Dest/route/carrier (incl. heavy smoothing)** — helps 2005 validation,
   hurts 2006 eval: the airport-level delay statistics drift between years.
2. **Monotone constraints on (adjusted) departure time, loss-guide growth, adjusted-time re-encoding**
   — all below the plain deep ensemble.
3. **More seeds (8/10), hyper-diverse or feature-diverse ensembles, heavier row bagging
   (subsample 0.5-0.6), max_bin 512, lr/rounds re-tuning** — all within noise of the final recipe.

## What I would try with more budget

The recipe has plateaued at ~0.759 on this 100k-row training slice, and the plateau is
information-limited rather than tuning-limited: the drift analysis shows the model gains almost all
of its power from stable time-of-day structure, so I would attack the drift itself. Concretely:
(1) time-decay/covariate-shift analysis per feature family — e.g., re-fit count stats per season to
reduce year drift, or shrink all learned structure toward global (time-only) effects; (2) a
proper 2-level stack: first-stage out-of-fold 2005 predictions to calibrate which feature families
transfer (a transfer-validation protocol that doesn't exist in the current setup); (3) AUC-oriented
training (rank:pairwise, feasible inside XGBoost) since the metric is rank-based, plus larger
ensembles only if CPU allows; (4) a small hyperparameter search coarser than what I ran (I expect
≤ +0.002); and (5) quantify the eval-vs-holdout slice difference by month to make sure the 2006
slices really are homogeneous — the biggest remaining risk is that the hidden slice 2 differs from
slice 1 in seasonal mix, in which case dropping the drifting features would pay off more than any
capacity increase.
