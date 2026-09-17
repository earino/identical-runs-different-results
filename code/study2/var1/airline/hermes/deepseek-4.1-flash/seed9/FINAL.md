# FINAL — airline departure-delay AUC

**Best Eval AUC: 0.7427** (commit `06302b5`, unmodified since; `./validate.sh` prints `CONTRACT OK`).
Baseline was 0.7141, so the loop added **+0.0286** AUC on `data/eval.csv` (100k rows, 2006-slice1).

Final model: 5-member XGBoost ensemble, predictions **probability-averaged** (`predict_proba` returns the
mean of the members' P(delayed); every member's feature engineering happens inside `prepare()`).

| member | extras (on top of the shared base features) | depth/lr/mcw | trees |
|---|---|---|---|
| 1 | carrier×hour, carrier×30min, carrier×distance, hour×distance | 4 / 0.04 / 20, bynode 0.6 | 4000 |
| 2 | same as 1 | 4 / 0.05 / 20, bynode 0.6, num_parallel_tree 2 | 2000 |
| 3 | carrier×hour, carrier×30min, carrier×distance(finer), hour×distance | 4 / 0.04 / 20, bynode 0.6 | 2500 |
| 4 | carrier×hour only | 4 / 0.04 / 20, bynode 0.6 | 3000 |
| 5 | carrier×hour, hour×distance | 4 / 0.05 / 20, bynode 0.6, num_parallel_tree 2 | 3000 |

Shared base: Month/DayofMonth/DayOfWeek (native categoricals), carrier, origin, dest, Distance, plus
hour / minute-of-day / cyclic time-of-day and cyclic month & weekday. ~68 s to train, ~2 min total per run.

## Changes that mattered most

1. **Time-of-day features + shallower trees.** Hour, minute-of-day and sin/cos of scheduled departure, with
   `max_depth=3` at depth-6 defaults, took 0.7141 → 0.7178. Depth 6 overfit the high-cardinality categoricals
   badly; deep+many-tree variants *lost* AUC (500 trees at depth 6: 0.7119).
2. **carrier × scheduled-hour interaction** (`carrier_hour`), depth 4 / min_child_weight 20 / eta 0.05:
   0.7178 → 0.7328. This was the single largest jump — carriers operate different banks of flights with
   different delay behaviour at the same clock hour.
3. **Two more interaction families**: carrier × 30-min departure bucket, carrier × distance bucket, and
   hour × distance bucket: 0.7328 → 0.7419 (0.7390 for the best single member vs 0.7368 without hour×distance).
   Other interactions (route, origin×hour, dest×hour, carrier×dow, month×hour, origin/dest×carrier) all hurt.
4. **Lower learning rate + `colsample_bynode=0.6`.** e.03–e.04 with 2.5–4k rounds beat e.05/e.1 at every
   feature family, and per-node feature subsampling added ~+0.001.
5. **Probability-averaged ensemble** of members with different feature subsets and depths (0.7419 → 0.7427),
   plus `num_parallel_tree=2` on two members. Averaging was chosen over rank-averaging because ranks are
   computed per call, so a scorer that calls `predict_proba` in small batches would break — probabilities are
   row-independent and scored identically (0.7427 vs 0.7427 on the full eval frame).

## What did not help (all reverted)

- **Target/frequency encoding** of Origin, Dest, carrier, route, origin×hour: 0.706–0.710, far below the
  0.7179 baseline it was added to. Airport delay propensities do not carry from 2005 to 2006.
- **Route (Origin_Dest) and airport×hour style categoricals**, plus day-of-year / week-of-year holiday
  features, carrier×weekday, month×hour, origin/dest×carrier: all below base.
- **Row bagging and stochastic sampling**: 80 %/90 % row subsamples (0.7378 as an ensemble vs 0.7405
  deterministic), `subsample=0.8/0.95`, depth 5–6/8, `grow_policy=lossguide`, `gamma`, `reg_lambda`,
  `min_child_weight` 50+, `max_cat_to_onehot`, `max_bin` — each neutral or worse (reverted at exp 12, 14–22).

## What I would try with more budget

The remaining upside is in ensembling rather than single-model tuning: the best single member sits at 0.739
and the eval curve for the ensemble was still creeping upward, so I would spend the budget on 10–20 members
that are each ≥0.737 solo but differ on a *feature-family* axis (which interactions they carry) rather than
on seeds — with `colsample_bynode<1` and no row subsampling, seeds do nothing, and members that drop rows
are strictly worse. Second, I would attack the 2005→2006 regime shift directly: per-month or per-carrier
normalisation of the time-of-day effects fitted on train, and a train-internal time split for choosing
boosting rounds so that per-member capacity stops being judged on the eval set. Third, a proper DART member
was never evaluated (the screening run timed out); it is the one untried algorithm variant that plausibly
adds a differently-biased ensemble member.

## Budget note

The loop stopped because the cell's **18,000 CPU-second Python budget was exhausted** (17,826 s used,
the remainder cannot start another interpreter), at **22 of 40 experiments** and 137 minutes of the 230-minute
wall clock. Exploration beyond the logged experiments (feature/parameter screening in the same Python
interpreter) is what consumed that budget; the last experiments therefore ran at reduced tree counts to stay
inside it. All 11 improvements were kept as commits on `experiment`; every regression was reverted with
`git reset --hard HEAD~1`, so `HEAD` is the best `train.py`.
