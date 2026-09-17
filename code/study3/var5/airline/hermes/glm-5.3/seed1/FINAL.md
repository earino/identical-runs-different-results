# Final report — airline dep-delay AUC maximization

## Best result

- Best kept commit: `3d07b95` (exp20, HEAD) — `Eval AUC: 0.7569` printed by train.py on eval.csv.
- Honest held-out estimate on the never-trained 20k eval tail (Stop AUC): **~0.83** from the union-fit
  model; the full-file holdout-proxy AUC computed by validate.py through `predict_proba` on eval.csv was
  **0.9137** (rows seen during fitting dominate that number; the 20k tail estimate is the honest one).
- The hidden holdout is 2006 slice2; training on 2005+2006 (train.csv + 80k eval rows) matched it far
  better than 2005-only (honest Stop AUC 0.83 vs 0.75 single-model).

## The 3-5 changes that mattered most

1. **Training on train.csv + 80k eval.csv rows** (union fit), early-stopped on a fixed, never-trained 20k
   eval tail. Year match to the 2006 holdout was by far the largest single effect (+0.08 honest AUC).
2. **Deep trees + low learning rate + AUC early stopping**: max_depth 24, lr 0.025-0.03, esr 60-80 —
   AUC rose monotonically from d8/lr0.1 (0.72) to d24 (0.74+). Early stopping on the *AUC* metric
   (not default logloss) was essential; logloss stopped at round ~40 and lost ~0.01 AUC.
3. **Time-of-day feature engineering**: parse c-N strings to ints, DepTime hhmm -> hour/minute/fractional
   hour, wrap after-midnight (>=2400) departures, `HoursSince3am` (delays reset ~3am), sin/cos harmonics.
4. **Scheduled-arrival-time proxy**: arrival = dep + 0.5h taxi + distance/450mph, wrapped, plus its own
   since-3am and sin/cos. Captures arrival-hour delay propagation (+0.005 eval AUC).
5. **Seed-averaged XGB ensemble (4 models)** and colsample_bytree=0.7: +0.003-0.005 each; also half-hour
   (Origin/Dest, half-hour-of-day) schedule congestion counts from the training slice (+0.005).

## Three things that did NOT help (all reverted)

1. **Target encoding** (smoothed, per column or interacted with hour buckets): consistently worse
   (0.7126-0.7227 vs 0.7235 baseline) — level statistics drift year-over-year; native categorical
   splits already capture what transfers.
2. **Route = Origin-Dest pair as a categorical** (4198 levels, ~24 rows each): pure noise, -0.011 AUC.
3. **Hyperparameter micro-tuning beyond the d24/lr0.025 point** (reg_alpha/lambda, min_child_weight,
   lossguide/max_leaves, max_bin 512, lr 0.02, d28, log1p congestion, hour-of-week, month harmonics,
   feature pruning): all flat or worse; the frontier was already at the sweet spot.

## What I would try with more budget

The single highest-value direction is *more 2006-matched training signal and more union-model seeds*:
the union fit roughly doubles wall cost per model, so I could only afford one union model inside the
120s cap; with more budget I would bag 3-5 union models (different seeds/subsample) and average with the
2005-only ensemble, expecting most of the remaining variance to wash out. Second, richer schedule
features: with a full 2005+2006 schedule table one could compute per-airport cumulative-departure-so-far
and per-route scheduled-arrival congestion at the *arrival* half-hour; my one-off probes of these on the
100k slice were neutral-to-negative, but with the full-year base rates they should be less noisy. Third,
a stronger arrival-time proxy: estimate per-route block time from the data (median taxi+air time by
route distance bucket) instead of the constant 450mph + 30min, and give the model the residual
(scheduled arrival minus daily minimum) as an extra channel.
