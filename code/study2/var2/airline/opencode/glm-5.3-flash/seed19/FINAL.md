# Final report — airline delay XGBoost

**Best Eval AUC: 0.7363** (baseline 0.7141, +0.0222). Final config: 3-member XGBoost ensemble
(max_depth 10/12/14, lr .02/.015/.01, per-member colsample_bytree .45/.5/.55, subsample .8, gamma 1,
reg_alpha 1, min_child_weight 20, reg_lambda 2, max_bin 512, native categoricals for Carrier/Origin/Dest),
each member early-stopped on eval.csv with eval_metric=auc, predictions probability-averaged.

## Changes that mattered most

1. **Early stopping on eval.csv with eval_metric="auc"** (+~0.014 over random-split ES): eval is 2006 data,
   same distribution as the hidden holdout; ES picks the iteration count for the shifted distribution and
   lets the model train on 100% of train.
2. **Feature engineering in prepare()** (+~0.002): parsing c-<n> strings to ints, DepTime → hour (raw 0–26
   plus mod-24), minute, cyclical sin/cos for month/dom/dow/time-of-day, log1p(Distance), red-eye/evening/
   quarter-minute flags, distance bins. DepTime is the dominant signal (delay rate ramps 0.04→0.82 across
   the day; DepTime values up to 2620 roll past midnight).
3. **Regularization sweep on the ensemble** (+~0.007 total): colsample_bytree 0.8→0.5 (best), gamma 0→1,
   then reg_alpha 1.0 (final +0.0007). Deeper trees (6→12) with slower lr (0.05→0.015) added ~+0.004.
4. **3-member diverse ensemble with per-member colsample** (+~0.001–0.002 over a single depth-12 model).

## Things that did not help

1. **Route (Origin×Dest) as a native categorical** — 4198 levels: 0.7107 vs 0.7286 in the same config (−0.018).
2. **Out-of-fold smoothed target encoding + count features** for carrier/origin/dest/route: 0.7207 vs 0.7216;
   native categorical splits already capture per-level effects at this sample size.
3. **OOF stacking** (3-fold fast model's prediction as a feature): 0.7354 vs 0.7356. Also: month×dow
   categorical + dom buckets (0.7267), interaction categoricals carrier×hour / origin×dow (timeout —
   native categorical handling is slow with many levels), lossguide member (0.7347), colsample_bylevel,
   mcw/lambda increases, subsample 0.7, deeper (14) single models, 4th/5th ensemble members.

## With more budget

- Larger seed-bagged ensembles need runtime headroom: training a 4th/5th member at max_bin 512 approaches
  the 120 s limit, so I would first cut eval-set scoring overhead (evaluate ES metric every 25 iterations
  via a manual loop) to fit 6–8 diverse members.
- Bayesian/Optuna-style search over (depth, lr, colsample, gamma, alpha, max_bin) per member around the
  found optimum, plus per-member max_bin/subsample diversity, which showed promise but never beat noise.
- A careful revisit of DepTime: finer time-of-day features (e.g., scheduled-vs-slot interactions,
  hour×dow smoothing) — the delay-vs-hour ramp is by far the strongest signal in this dataset.
