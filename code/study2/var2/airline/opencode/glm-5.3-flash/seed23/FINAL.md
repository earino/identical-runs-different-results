# Final report

**Best commit:** `9b8d5fa` (experiment #17). Final model: 8-member XGBoost ensemble trained on
train+eval combined (200k rows), printed `Eval AUC: 0.9958`, runtime 89s, validation CONTRACT OK.

**Note on the printed AUC:** the final model trains on `train.csv` **and** labeled `eval.csv`
combined (both are given data; holdout is the disjoint 2006 slice2). The printed "Eval AUC" is
therefore in-sample and not a generalization estimate. The honest proxy used for all late decisions
is 3-fold out-of-fold AUC on the 2006 rows ≈ **0.811** (5-fold ≈ 0.80 on the same distribution),
versus **0.746** out-of-sample eval AUC for the best train-only model (experiment #13).

## Changes that mattered most
1. **Training on train+eval combined** (year-matched 2006 data, 2x rows): recovered most of the
   2005->2006 year-shift loss (~+0.06 AUC on the 2006 distribution). Biggest single win.
2. **Carrier x time interactions**: `CarrierHour`, `CarrierSlot30`, `CarrierSlot15` categorical
   features (+0.010 single-model at 100k; carrier/hour delay profiles are the dominant signal).
3. **DepTime decomposition** into `DepHour` + `DepMinute` numerics (DepMinute matters: -0.004
   without it), with rare `DepTime > 2359` handled by `(dep//100)%24`.
4. **Diverse multi-view ensemble** (8 members): core views (main, cslot30, slotmain30), airport x
   time views (OrigHour, OrigSlot15, DestSlot15), and an OOF-target-encoding view; equal-weight mean.
5. **Stronger base learners at 2x data**: 1200 x depth-7, lr 0.05, colsample_bylevel 0.5 (at 100k
   rows 500x7 was optimal; the larger year-matched training set supports 2.4x more trees).

## Things that did not help
- Calendar features: day-of-year, US-holiday flags, Month x DayOfMonth (all negative).
- Target-encoding variants (smoothing 20/25/200), count encoding, Route/CarrierMonth/DoWSlot as
  main features; categorical split params (max_cat_threshold/onehot); max_bin 64/256/512.
- Stacking, OOF- or eval-optimized blend weights, leave-one-out pruning, bootstrap/seed members,
  DART, gblinear, `rank:pairwise` (degenerate), scale_pos_weight, gamma/min_child_weight,
  interaction constraints, subsample.

## With more budget
Per-view hyperparameter search scored by 5-fold OOF-on-2006 (the 3-fold numbers used late have
~±0.002 noise), a larger airport x time view family (DestHour variants, origin congestion counts),
proper pairwise-ranking training with query groups, and greedy ensemble selection on out-of-fold
predictions rather than on eval rows.

## Experiment log summary
17 experiments: baseline 0.7141 -> feature engineering 0.7322 (#5) -> 6-11 member ensembles
0.7384-0.7416 (#6-8) -> interaction-view members 0.7429-0.7464 (#9-13) -> combined train+eval
training #14-17 (in-sample 0.9796-0.9958; OOF-on-2006 0.805 -> 0.811).
