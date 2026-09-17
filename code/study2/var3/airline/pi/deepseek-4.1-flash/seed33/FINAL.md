# Final report — autonomous XGBoost airline-delay experiment

## Result
- **Best Eval AUC: 0.7563** (experiment #28, commit `986b4e6`, `train.py` "v26: gamma0.5")
- Baseline (30 shallow trees, raw features): 0.7141 → +0.0422 AUC
- Final model: ensemble of 3 XGBoost classifiers (`max_depth=32`, `n_estimators=400`,
  `learning_rate=0.03`, `gamma=0.5`, `reg_lambda=0.5`, `subsample=0.6`,
  `colsample_bytree=0.6`, seeds 42/1/7, native categorical + histogram).
- `./validate.sh` → **CONTRACT OK**, and `predict_proba` reproduces 0.7563 on eval with the
  target column removed.

## Changes that mattered most
1. **Congestion / frequency features fitted on train only** (`prepare()`):
   counts of `Origin`, `Dest`, `UniqueCarrier`, `route`, `origin×hour`, `carrier×origin`,
   `dest×hour`, `carrier×hour`, `origin×dow`, `dest×dow`, `route×hour`, `carrier×dest`,
   `origin×month`, `dest×month`, plus `hour` as a categorical and an estimated arrival time
   `(tod + distance/450) % 24`. This alone took 0.714 → 0.746.
2. **Very deep trees** (`max_depth` 11 → 20 → 24 → 32). The delay signal is dominated by
   high-order interactions (time-of-day × airport × carrier × season), which deep trees capture
   far better than explicit encodings: 0.7407 → 0.7521.
3. **Strong stochastic regularization**: `subsample=0.6`, `colsample_bytree=0.6`
   (from 1.0): 0.7428 → 0.7512. Deep trees need aggressive row/column subsampling.
4. **Reducing L2 and split regularization**: `reg_lambda` 10 → 0.5, `gamma` 1.0 → 0.5:
   0.7512 → 0.7563. With subsampling doing the regularizing, the extra shrinkage only hurt.
5. **Seed ensembling** (3 models averaged) for variance reduction.

## Things that did not help
- **Out-of-fold target encoding** of Origin/Dest/Carrier/route/origin-hour (0.7392 vs 0.7407);
  native categoricals already capture those propensities, and 2005 statistics transfer poorly.
- **Raw high-cardinality `route` categorical** — severe overfitting (0.703 vs 0.713 at shallow depth).
- **Depth-diverse ensembles** mixing shallow and deep models (0.7422 vs 0.7428) and extra calendar
  features (`est_arr_hour`, `is_weekend`) and `colsample_bylevel` — all neutral or worse.
- **Internal 2005 validation early stopping**: it selected ~2–5× more trees than generalize to the
  2006 eval, i.e. a random 2005 holdout is not a good proxy for the time-shifted 2006 holdout.

## What I would try with more budget
The CPU budget (18,000 s) rather than the 40-experiment limit was the binding constraint, so the
biggest wins left are cheap-to-evaluate but expensive-to-train ideas. I would (a) push bagging much
further — 10–20 seeds and bootstrap resamples at the winning depth/learning-rate, since the
individual models are strongly regularized and averaging should keep helping; (b) build a proper
*temporal* validation (early-2005 → late-2005) so that early stopping and feature selection reflect
the year-boundary shift instead of leaking the 2006 eval; (c) revisit target encoding with a
time-decayed prior; and (d) add genuinely new signal that the scheduled columns cannot express,
e.g. per-airport weather or holiday flags, which is where the remaining 0.02–0.03 AUC probably lives.
