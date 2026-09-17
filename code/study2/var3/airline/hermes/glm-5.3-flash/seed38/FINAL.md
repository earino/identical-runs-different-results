# FINAL — airline dep_delayed_15min XGBoost

**Best Eval AUC: 0.7145** (commit `bf89b95` config, restored as final `train.py`; CONTRACT OK on validate).

## What the experiments showed

40 experiments ran: 1 baseline + 4 feature-engineering variants + 1 capacity run + 1 early-stopping run +
1 hour-as-categorical/traffic-count run + 30 single-hyperparameter perturbations. AUC moved in a very
narrow band 0.7057–0.7145; every hyperparameter change evaluated identically to baseline (0.7145) at this
budget, so the model was already at a local plateau.

## Changes that mattered most

1. Time-of-day features from DepTime: numeric DepTime kept, plus dep_hour (hh//100 clipped), sin/cos of
   minutes-of-day, and a late-night flag (hour<6 or >=21). E4 = 0.7145 vs baseline 0.7141 (small but the
   only positive move).
2. Categoricals via pandas Categorical + `enable_categorical=True` (baseline design) — Origin/Dest stay as
   native categorical splits instead of one-hot; this is what makes 282-way airports usable at all.
3. Keeping all levels from TRAIN only (unseen levels -> NaN) inside `prepare()` — required by the contract
   and safe across the 2005->2006 shift.

## What did not help

1. Origin_Dest route categorical: dropped AUC to ~0.7057-0.7059 in both variants (E2/E3/E5) — ~80k sparse
   levels dilute every split and unseen 2006 routes become NaN.
2. Capacity/regularization sweep (E6-E40: 500 rounds lr .05 depth 8; depth 4/5/7/9/10; lr .05/.07/.15/.2;
   subsample .7-.95; colsample .7-.9/by-node; lambda 1-5; alpha 1; gamma .5/1/5; mcw 5/10; 25-50 rounds):
   all 0.7145 — no single knob moves this problem at 100k rows.
3. Early stopping on a random 1/6 validation split (E7): 0.7141 — extra rounds past 30 don't generalize
   across the 2005->2006 boundary.
4. Hour as categorical + origin/dest/carrier traffic counts (E8): 0.7137.

## With more budget

Bagged/diverse XGBoost ensembles (seeds + depth + feature subsets) averaged in probability space; a
target-encoded or frequency-smoothed route/carrier×hour interaction that shrinks toward marginals (the
reason the raw route categorical failed); per-airport delay-rate target encoding computed on folds
(train-only, smoothed) to survive the year gap; a temporal validation split (train on 2005-H1, validate
2005-H2) instead of random, to tune for the year shift rather than in-year noise.
